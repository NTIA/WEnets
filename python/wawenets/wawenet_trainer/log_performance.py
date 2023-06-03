from typing import Dict, Union

import torch

import numpy as np
import pytorch_lightning as pl


def log_performance_metrics(
    outputs: dict, pl_module: pl.LightningModule, phase: str
) -> dict:
    """
    processes outputs from train/test/val batches and epochs and generates performance
    measurements.

    Parameters
    ----------
    outputs : dict
        outputs from `LitWAWEnetModule.train_step`, `LitWAWEnetModule.test_step`,
        or `LitWAWEnetModule.val_step`. should at minimum contain a `y` and
        `y_hat` field.
    pl_module : pl.LightningModule
        an instance of `LitWAWEnetModule`
    phase : str
        the phase in the training process to which the outputs belong

    Returns
    -------
    dict
        contains the loss and correlatons calculated from the input data.
    """
    # if we've gotten all batch outputs (instead of a single batch output),
    # IE at the end of an epoch, stack. only wanna stack this stuff once
    if isinstance(outputs, list):
        y = torch.vstack([item["y"] for item in outputs])
        y_hat = torch.vstack([item["y_hat"] for item in outputs])
    else:
        y = outputs["y"]
        y_hat = outputs["y_hat"]
    loss = pl_module.loss_fn(y_hat, y)
    pl_module.log(f"{phase} loss".replace(" ", "_"), loss)
    correlations = _log_correlations(y, y_hat, pl_module, phase)
    return {"loss": loss, "correlations": correlations}


def calculate_correlation(
    y: Union[torch.tensor, np.ndarray], y_hat: Union[torch.tensor, np.ndarray]
) -> np.float64:
    """
    thin wrapper around `np.corrcoef`—grabs the relevant element from a
    correlation matrix. assumes inputs are one-dimensional.

    Parameters
    ----------
    y : Union[torch.tensor, np.ndarray]
        truth-data values for a given target.
    y_hat : Union[torch.tensor, np.ndarray]
        corresponding prediction values for a given talget

    Returns
    -------
    np.float64
        the pearson's correlation of the 1-d input data
    """
    corr_matrix = np.corrcoef(y, y_hat)
    return corr_matrix[0, 1]


def _log_correlations(
    y: torch.tensor,
    y_hat: torch.tensor,
    pl_module: pl.LightningModule,
    phase: str,
) -> Dict[str, np.float64]:
    """
    calculates individual and mean pearson correlations for given target and
    predicted values.

    Parameters
    ----------
    y : torch.tensor
        truth-data values for a given target
    y_hat : torch.tensor
        corresponding prediction values for a given target
    pl_module : pl.LightningModule
        instance of `LitWAWEnetModule
    phase : str
        current phase in the training process, either train, test, or validation

    Returns
    -------
    Dict[str, np.float64]
        mean and per-target correlations for the input data
    """
    # number of columns should match number of targets
    assert y.shape[1] == len(pl_module.normalizers)
    correlations = dict()
    all_pearson_r = np.zeros((0))
    for ind, normalizer in enumerate(pl_module.normalizers):
        pearson_r = calculate_correlation(y[:, ind], y_hat[:, ind])
        correlations[normalizer.name] = pearson_r
        if "batch" in phase:
            pl_module.log(
                f"{phase} correlation {normalizer.name}".replace(" ", "_"), pearson_r
            )
        all_pearson_r = np.hstack((all_pearson_r, pearson_r))
    mean_pearsor_r = all_pearson_r.mean()
    pl_module.log(f"{phase} correlation".replace(" ", "_"), mean_pearsor_r)
    correlations["mean"] = mean_pearsor_r
    return correlations
