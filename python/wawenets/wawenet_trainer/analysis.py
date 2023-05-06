from io import BytesIO
from typing import Dict, List, Tuple, Union

import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import colors
from sklearn.metrics import mean_absolute_error as mae

from wawenet_trainer.lightning_model import LitWAWEnetModule
from wawenet_trainer.log_performance import (
    calculate_correlation,
    log_performance_metrics,
)
from wawenet_trainer.transforms import NormalizeGenericTarget

"""do all the test data post processing here.

this is part analysis and part record keeping. we will end up logging
the results of these analysis to clearML, which we can query later on
and do any further postprocessing we desire.

when we're evaluating model performance, we get this information for each
test batch/step (this comes from LitWAWEnetModule.test_step):
 {
    step loss
    y
    y_hat
    df_ind
    language
    impairment
 }
we can think of this as a list of records from which we can make a dataframe. then
we can use `df.groupby` to handle per-condition and per-language calculations while
sharing code.

TODO: will need to make a mechanism for remapping group names, esp.
      for the original ITS dataset
"""


class WENetsAnalysis:
    """makes a dataframe from the outputs we get at the end of a testing epoch
    and performs analysis using that dataframe

    specifically, `LitWAWEnetModule.test_step` returns a dictionary of info for each batch.
    these dictionaries are concatenated into a list and that list is accessible in
    `LitWAWEnetModule.test_epoch_end` and is stored as a class attribute.
    `WAWEnetCallbacks.on_test_epoch_end` accesses that attribute and uses it to
    initialize an instance of `WeNetsAnalysis`.

    supports callbacks.WAWEnetCallbacks"""

    def __init__(
        self,
        test_outputs: List[dict],
        pl_module: LitWAWEnetModule,
        dataloader_name: str,
    ) -> None:
        """
        `WENetsAnalysis init fn`

        Parameters
        ----------
        test_outputs : List[dict]
            model predictions coming from `LitWAWEnetModule.test_epoch_end`
        pl_module : LitWAWEnetModule
            provides access to loging, normalizations when we need them
        dataloader_name : str
            name of the dataloader used to generate model predictions
        """
        # we have to pass the pl_module around because that's how things are
        # actually logged
        self.test_outputs = test_outputs
        self.pl_module = pl_module
        self.dataloader_name = dataloader_name
        # gotta stack all the outputs before we can make a DF.
        # but to do that, we need all the fields
        fields = test_outputs[0].keys()
        # make a dictionary with all field data stacked
        stacked = {
            field_name: self._stack_field(test_outputs, field_name)
            for field_name in fields
        }

        # explode and denormalize the target columns in the y/yhat tensors
        # explode: each column contains predictions for a distinct target.
        #          here we put them into their own field in the dictionary.
        # denormalize: the predictions are still normalized to [-1, 1]. here
        #           we put them back into the target's native range.
        for index, normalizer in enumerate(pl_module.normalizers):
            stacked[normalizer.name] = normalizer.denorm(stacked["y"][:, index])
            stacked[f"{normalizer.name}_hat"] = normalizer.denorm(
                stacked["y_hat"][:, index]
            )
        # delete y/y_hat, otherwise pandas will complain when we make a DF
        del stacked["y"]
        del stacked["y_hat"]
        # delete test_step_loss because its length is number of steps, not number
        # of samples
        del stacked["test_step_loss"]

        # make a dataframe from that dictionary
        self.df = pd.DataFrame.from_dict(stacked)

        # do some matplotlib init. there's got to be a better way to do this
        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=22)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rc("font", family="Times New Roman")
        plt.rc("text", usetex=True)

    @staticmethod
    def _stack_field(
        test_outputs: List[dict], field_name: str
    ) -> Union[torch.Tensor, list]:
        """
        traverses a list of dictionaries. extracts the lists or tensors
        present in the dictionary and stacks them into a single list or tensor.

        Parameters
        ----------
        test_outputs : List[dict]
            the outputs available in `LitWAWEnetModule.test_step_outputs` after
            completion of a test epoch
        field_name : str
            the name of the field in each dictionary that contains a list
            or tensor that should be stacked.

        Returns
        -------
        Union[torch.Tensor, list]
            all of the data contained in each dictionary with the field `field_name`,
            stacked into a single tensor or list.
        """

        # extracts lists/tensors from an iterable of dictionaries and concatenates
        # them.

        # List[Dict[str: any]] -> Union[List, Tensor]
        if isinstance(test_outputs[0][field_name], torch.Tensor):
            if len(test_outputs[0][field_name].shape) == 1:
                # handle any tensors that are 1D
                return torch.hstack([item[field_name] for item in test_outputs])
            else:
                # handle 2D tensors, including those with a singleton second-dimension
                return torch.vstack([item[field_name] for item in test_outputs])
        else:
            stacked = list()
            for item in test_outputs:
                stacked.extend(item[field_name])
            return stacked

    def log_performance_metrics(self, dataloader_name: str):
        return log_performance_metrics(
            self.test_outputs, self.pl_module, dataloader_name
        )

    def _generate_performance_record(
        self, normalizer_name: str, df: pd.DataFrame
    ) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        generate a dictionary containing performance metrics for
        a specific target

        Parameters
        ----------
        normalizer_name : str
            the name of the target for which metrics should be extracted
        df : pd.DataFrame
            a dataframe in the format generated by self.__init__

        Returns
        -------
        Tuple[dict, np.ndarray, np.ndarray]
            a performance record dictionary, target values, predicted values
        """
        # generate a dictionary containing performance metrics for a specific target
        performance_record = dict()
        y = df[normalizer_name].to_numpy()
        y_hat = df[f"{normalizer_name}_hat"].to_numpy()
        performance_record["target"] = normalizer_name
        performance_record["samp"] = len(df)
        performance_record["corr"] = calculate_correlation(y, y_hat)
        # converting to tensors and then back to numpy is a little painful but this way
        # we get to use the same loss function that we used to train/test.
        loss = (
            self.pl_module.loss_fn(
                torch.Tensor(y),
                torch.Tensor(y_hat),
            )
            .numpy()
            .item()
        )
        performance_record["loss"] = loss
        performance_record["mae"] = mae(y, y_hat)
        performance_record["tavg"] = y.mean()
        performance_record["pavg"] = y_hat.mean()
        return performance_record, y, y_hat

    def _per_target_performance_records(
        self, df: pd.DataFrame, group_name: str = None, scatter_plot: bool = False
    ) -> List[dict]:
        """
        can be used to generate performance records for a specified group
        or for aggregated results

        Parameters
        ----------
        df : pd.DataFrame
            a dataframe in the format generated by self.__init__
        group_name : str, optional
            if `df` has been grouped by a column, provide the name of that group
            here, by default None
        scatter_plot : bool, optional
            whether or not to generate a scatter plot of target vs. predicted
            values, by default False

        Returns
        -------
        List[dict]
            a list containing performance record dictionaries. performance record
            dictionaries contain information about the data and predictions
            associated with the target.
        """

        # generate performance metrics for all targets—this works be cause each target
        # has a column for target values and predictions in the DF—the result of "exploding"
        # in `__init__`
        performance_records = list()
        for normalizer in self.pl_module.normalizers:
            performance_record, y, y_hat = self._generate_performance_record(
                normalizer.name, df
            )
            # if this `df` only contains data from a specific group, keep track of that here
            if group_name:
                performance_record["group"] = group_name
            performance_records.append(performance_record)
            # if we've been instructed to make a scatter plot, do it here.
            if scatter_plot:
                self._twod_hist(
                    normalizer,
                    y,
                    y_hat,
                    performance_record["corr"],
                    performance_record["loss"],
                    cmap=self.pl_module.scatter_color_map,
                )

        return performance_records

    def target_performance_metrics(self) -> pd.DataFrame:
        """
        used to generate overall results

        Returns
        -------
        pd.DataFrame
            a pandas dataframe where each row has columns that contain
            performance information for a given target.
        """
        # used to generate overall results
        performance_records = self._per_target_performance_records(
            self.df, scatter_plot=True
        )
        return pd.DataFrame(performance_records)

    def grouped_performance_metrics(self, group_column: str) -> pd.DataFrame:
        """
        generate performance metrics for all targets, but also grouped by
        the values in `group_column`

        Parameters
        ----------
        group_column : str
            the name of the column that should be used to group rows
            of the dataframe.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe with num_groups x num_targets rows, where
            each row contains performance information for group by target
            combinations.
        """
        # this generates a new `df` which can then also be processed,
        # e.g., by `per_condition_metrics`
        grouped_performance_records = list()
        # loop over all possible groups
        for group_name, gdf in self.df.groupby(by=group_column):
            # loop over all our targets
            # make a list of dictionaries and then make a dataframe from that
            performance_records = self._per_target_performance_records(
                gdf, group_name=group_name
            )
            grouped_performance_records.extend(performance_records)
        group_df = pd.DataFrame(grouped_performance_records)
        return group_df

    def per_condition_metrics(self, grouped_df: pd.DataFrame) -> pd.DataFrame:
        """
        calculates "per condition" performance metrics. in short, losses and
        correlations are calculated for targets and predictions on each "group",
        and then a loss and correlation is calculated on the resulting values.
        assumes that `grouped_df` is an output from `grouped_performance_metrics`
        and that `group_column` represents an impairment of some kind.

        Parameters
        ----------
        grouped_df : pd.DataFrame
            a dataframe generated by `grouped_performance_metrics`

        Returns
        -------
        pd.DataFrame
            a pandas dataframe with one row per target and columns
            `per_condition_correlation` and `per_condition_loss`
        """
        # group by target, then do correlation and loss for each
        per_condition_metrics = list()
        for group_name, gdf in grouped_df.groupby(by="target"):
            record = {"target": group_name}
            record["per_condition_correlation"] = calculate_correlation(
                gdf["tavg"], gdf["pavg"]
            )
            record["per_condition_loss"] = (
                self.pl_module.loss_fn(
                    torch.Tensor(gdf["tavg"].to_numpy()),
                    torch.Tensor(gdf["pavg"].to_numpy()),
                )
                .numpy()
                .item()
            )
            per_condition_metrics.append(record)
        per_condition_df = pd.DataFrame(per_condition_metrics)
        return per_condition_df

    def _twod_hist(
        self,
        normalizer: NormalizeGenericTarget,
        y: np.ndarray,
        y_hat: np.ndarray,
        correlation: float,
        loss: float,
        cmap: str = "Greys",
    ):
        """
        generates a 2d histogram of target value vs. predicted value.
        like a scatter plot, except you can see what's happening when
        there's lots of overlap.

        Parameters
        ----------
        normalizer : NormalizeGenericTarget
            the normalizer for the target that represents the `y` and
            `y_hat` data being provided.
        y : np.ndarray
            target values
        y_hat : np.ndarray
            predicted values
        correlation : float
            the correlation between `y` and `y_hat`. passed in here so we only
            calculate it once.
        loss : float
            the loss between `y` and `y_hat`.
        cmap : str, optional
            the matplotlib colormap name to be used to provide false color
            for the 2d histogram, by default "Greys"
        """
        fig, ax = plt.subplots(dpi=300, figsize=(6, 5))
        beg, end, step = normalizer.tick_start_stop_step
        x_start, x_stop = normalizer.MIN, normalizer.MAX + 0.1
        y_start, y_stop = normalizer.MIN, normalizer.MAX + 0.1

        # figure out if we need to show the histogram on log scale
        if len(y.squeeze()) < 3000:
            norm_fun = colors.Normalize()
        else:
            norm_fun = colors.LogNorm()

        hax = ax.hist2d(
            y.squeeze(),
            y_hat.squeeze(),
            bins=100,
            norm=norm_fun,
            cmap=cmap,
            zorder=3,
        )

        ax.plot([x_start, x_stop], [y_start, y_stop], "--", zorder=4, color="0.8")

        ax.set_xlabel(r"\textbf{target}")
        ax.set_xbound([x_start, x_stop])
        ax.set_ybound([y_start, y_stop])
        ax.set_xticks(np.arange(beg, end, step))
        ax.set_yticks(np.arange(beg, end, step))

        ax.axis([x_start, x_stop, y_start, y_stop])
        ax.annotate(
            r"$\rho_{seg}" + f"={correlation:.3f}" + r"$",
            xy=(0.600, 0.21),
            xycoords="axes fraction",
            fontsize="large",
        )
        ax.annotate(
            r"$\textrm{RMSE}=" + f"{loss:.3f}" + r"$",
            xy=(0.520, 0.12),
            xycoords="axes fraction",
            fontsize="large",
        )
        ax.axis("equal")

        ax.grid(zorder=-1)
        cb = fig.colorbar(hax[3], ax=ax, extend="max")
        cb.outline.set_visible(False)

        ax.set_ylabel(r"\textbf{predicted target}")

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # this feels dirty: report the matplotlib fig all the way in here
        #
        # commenting for now because it didn't work. it felt dirty anyway.
        # maybe i can just return the fig and then report it from the callback.
        #
        plt.subplots_adjust(wspace=0.05)
        self.pl_module.clearml_task.logger.report_matplotlib_figure(
            title=f"{normalizer.name}", series=f"{self.dataloader_name}", figure=plt
        )

        # fig.savefig(
        #     f"{self.pl_module.output_uri / 'uploads'}/{self.dataloader_name}_{normalizer.name}.jpg"
        # )

        # TODO: hmm, how do we just save a plot to disk if we aren't logging
        #       to clearML :(
        #       seems like we can check if the task is None, and if it is,
        #       come up with a name and just write it directly to the output_uri
        # TODO: does `report_media` make it possible to get the PDFs via the API?
        # TODO: why are all the graphs green? that's not right
        # report_media
        with BytesIO() as buffer:
            fig.savefig(
                buffer,
                format="pdf",
                bbox_inches="tight",
            )
            self.pl_module.clearml_task.logger.report_media(
                f"{self.dataloader_name}_{normalizer.name}",
                series=f"{self.dataloader_name}",
                stream=buffer,
                file_extension="pdf",
                iteration=self.pl_module.global_step,
            )
        # # report matplotlib figure
        # it's nice to have this for quick looking in clearML, even if the JPEG
        # you download from clearML is blank.
        self.pl_module.clearml_task.logger.report_matplotlib_figure(
            f"{self.dataloader_name}_{normalizer.name}",
            series=f"{self.dataloader_name}",
            iteration=self.pl_module.global_step,
            figure=fig,
        )
