from typing import List

import torch
import pytorch_lightning as pl

import numpy as np
import pandas as pd

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
1. per_cond_df
2. result vectors for each target
3. result vectors for each target, by condition (?)
    this might be a bit much?
4. `test_results_table`, make it a DF, and use pandas to write out markdown tables
    columns: samples, correlation, rmse, mae, tavg, pavg
5. `test_grouped_results_table`, make it a DF and use pandas to write out markdown tables
6. training and validation correlation and loss at the end of each epoch.
    but maybe not because this is already a scalar graph, and i think we can 
    get at that data programmatically using the clearML api.
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


def _calculate_correlation(y: torch.tensor, y_hat: torch.tensor):
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
    def __init__(self, test_outputs: List[dict], pl_module: pl.LightningModule) -> None:
        self.test_outputs = test_outputs
        self.pl_module = pl_module
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
    def _stack_field(test_outputs: List[dict], field_name: str):
        if isinstance(test_outputs[0][field_name], torch.Tensor):
            if len(test_outputs[0][field_name].shape) == 1:

                return torch.hstack([item[field_name] for item in test_outputs])
            else:
                return torch.vstack([item[field_name] for item in test_outputs])
        else:
            stacked = list()
            for item in test_outputs:
                stacked.extend(item[field_name])
            return stacked

    def log_performance_metrics(self):
        return log_performance_metrics(self.test_outputs, self.pl_module, "test")

    def grouped_performance_metrics(self, group_column: str):
        group_performance = dict()
        # loop over all possible groups
        for group_name, gdf in self.df.groupby(by=group_column):
            # loop over all our targets
            # TODO: better data structure, and don't overwrite previous target's results
            for normalizer in self.pl_module.normalizers:
                group_performance[group_name] = self.pl_module.loss_fn(
                    torch.Tensor(gdf[normalizer.name].to_numpy()),
                    torch.Tensor(gdf[f"{normalizer.name}_hat"].to_numpy()),
                )
        print(group_performance)
