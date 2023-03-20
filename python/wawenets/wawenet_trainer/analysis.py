from io import BytesIO
from typing import List, Tuple, Union

import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import colors
from sklearn.metrics import mean_absolute_error as mae

from wawenet_trainer.lightning_model import LitWAWEnetModule
from wawenet_trainer.transforms import NormalizeGenericTarget

"""do all the test data post processing here.

this is part analysis and part record keeping. we will end up logging
the results of these analysis to clearML, which we can query later on
and do any further postprocessing we desire.

when we're evaluating model performance, we get this information for each
test batch/step:
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

items to consider reporting:
1. DONE--per_cond_df
2. result vectors for each target
3. result vectors for each target, by condition (?)
    this might be a bit much?
4. DONE--`test_results_table`, make it a DF, and use pandas to write out markdown tables
    columns: samples, correlation, rmse, mae, tavg, pavg
5. DONE--`test_grouped_results_table`, make it a DF and use pandas to write out markdown tables
6. DONE--training and validation correlation and loss at the end of each epoch.
    but maybe not because this is already a scalar graph, and i think we can 
    get at that data programmatically using the clearML api.

TODO: will need to make a mechanism for remapping group names, esp.
      for the original ITS dataset
"""


def log_performance_metrics(
    outputs: dict, pl_module: pl.LightningModule, phase: str
) -> dict:
    # if we've gotten all batch outputs (instead of a single batch output),
    # IE at the end of an epoch, stack. only wanna stack this stuff once
    if isinstance(outputs, list):
        y = torch.vstack([item["y"] for item in outputs])
        y_hat = torch.vstack([item["y_hat"] for item in outputs])
    else:
        y = outputs["y"]
        y_hat = outputs["y_hat"]
    loss = pl_module.loss_fn(y_hat, y)
    pl_module.log(f"{phase} loss", loss)
    correlations = _log_correlations(y, y_hat, pl_module, phase)
    return {"loss": loss, "correlations": correlations}


def _calculate_correlation(
    y: Union[torch.tensor, np.ndarray], y_hat: Union[torch.tensor, np.ndarray]
):
    corr_matrix = np.corrcoef(y, y_hat)
    return corr_matrix[0, 1]


def _log_correlations(
    y: torch.tensor,
    y_hat: torch.tensor,
    pl_module: pl.LightningModule,
    phase: str,
):
    # number of columns should match number of targets
    assert y.shape[1] == len(pl_module.normalizers)
    correlations = dict()
    all_pearson_r = np.zeros((0))
    for ind, normalizer in enumerate(pl_module.normalizers):
        pearson_r = _calculate_correlation(y[:, ind], y_hat[:, ind])
        correlations[normalizer.name] = pearson_r
        if "batch" in phase:
            pl_module.log(f"{phase} correlation: {normalizer.name}", pearson_r)
        all_pearson_r = np.hstack((all_pearson_r, pearson_r))
    mean_pearsor_r = all_pearson_r.mean()
    pl_module.log(f"{phase} correlation", mean_pearsor_r)
    correlations["mean"] = mean_pearsor_r
    return correlations


class WENetsAnalysis:
    def __init__(
        self,
        test_outputs: List[dict],
        pl_module: LitWAWEnetModule,
        dataloader_name: str,
    ) -> None:
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
        for index, normalizer in enumerate(pl_module.normalizers):
            stacked[normalizer.name] = normalizer.denorm(stacked["y"][:, index])
            stacked[f"{normalizer.name}_hat"] = normalizer.denorm(
                stacked["y_hat"][:, index]
            )
        # delete y/y_hat, otherwise pandas will complain
        del stacked["y"]
        del stacked["y_hat"]
        # delete test_step_loss because its length is number of steps, not number
        # of samples
        del stacked["test_step_loss"]

        # make a dataframe from that dictionary
        self.df = pd.DataFrame.from_dict(stacked)

    @staticmethod
    def _stack_field(
        test_outputs: List[dict], field_name: str
    ) -> Union[torch.Tensor, list]:
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
        # generate a dictionary containing performance metrics for a specific target
        performance_record = dict()
        y = df[normalizer_name].to_numpy()
        y_hat = df[f"{normalizer_name}_hat"].to_numpy()
        performance_record["target"] = normalizer_name
        performance_record["samp"] = len(df)
        performance_record["corr"] = _calculate_correlation(y, y_hat)
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
        # generate performance metrics for all targets
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
        performance_records = self._per_target_performance_records(
            self.df, scatter_plot=True
        )
        return pd.DataFrame(performance_records)

    def grouped_performance_metrics(self, group_column: str) -> pd.DataFrame:
        # generate performance metrics for all targets, but also grouped by
        # the values in `group_column`
        #
        # this generates a new `df` which can then also be processed
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

    def per_condition_metrics(self, grouped_df: pd.DataFrame):
        # group by target, then do correlation and loss for each
        per_condition_metrics = list()
        for group_name, gdf in grouped_df.groupby(by="target"):
            record = {"target": group_name}
            record["per_condition_correlation"] = _calculate_correlation(
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
        # ax.axis('equal')

        ax.grid(zorder=-1)
        cb = fig.colorbar(hax[3], ax=ax, extend="max")
        cb.outline.set_visible(False)

        ax.set_ylabel(r"\textbf{predicted target}")

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # plt.subplots_adjust(wspace=0.05)
        # this feels dirty: report the matplotlib fig all the way in here
        self.pl_module.clearml_task.logger.report_matplotlib_figure(
            title=f"{normalizer.name}", series=f"{self.dataloader_name}", figure=plt
        )
        plt.show()

        # fig_save_path = plot_parent
        # if not fig_save_path.is_dir():
        #     fig_save_path.mkdir(exist_ok=True)
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
        )
        self.pl_module.clearml_task.logger.report_media(
            f"{self.dataloader_name}_{normalizer.name}",
            series=f"{self.dataloader_name}",
            stream=buffer,
            file_extension="png",
        )
