from typing import Any, List, Optional

import pytorch_lightning as pl

import torch

import numpy as np

from wawenet_trainer.transforms import NormalizeGenericTarget

# set up callbacks here. this is the tricky one, maybe


# we have to do some stupid gymnastics because:
# 1. `on_[training/validation]_batch_end` receive outputs from the model
# 2. `on_[training/validation]_epoch_end` don't receive outputs from the model
# 3. `[training/validation]_epoch_end` in the pl.Module DO receive outputs from the
#     model
#
# otherwise we'd have some nice separation


def _log_performance_metrics(
    outputs: dict, pl_module: pl.LightningModule, phase: str
) -> dict:
    # this will happen at the end of each batch.
    # because i only wanna stack this stuff once
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


class TestCallbacks(pl.Callback):
    def __init__(self, normalizers: List[NormalizeGenericTarget] = None) -> None:
        self.normalizers = normalizers
        # ugh, after all the work i did to be able to log in two places,
        # i don't see a way around keeping track of predictions and actual values.
        # this is pretty hairy tbh.
        self.y_train: list = list()
        self.y_hat_train: list = list()
        self.y_val: list = list()
        self.y_hat_val: list = list()
        self.y_test: list = list()
        self.y_hat_test: list = list()
        self.loss_dict = {
            "training": list(),
            "validation": list(),
        }
        self.corr_dict = {
            "training": list(),
            "validation": list(),
        }
        super().__init__()

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # TODO: maybe use torch.vstack instead of turning everything into a list?
        # TODO: make sure tensors don't remain on GPU
        # TODO: handle denormalization here, since this is where we're doing reporting
        # TODO: handle reporting by language, impairment here
        self.y_val = list()
        self.y_hat_val = list()
        return super().on_validation_start(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,  #: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        performance_metrics = _log_performance_metrics(
            outputs, pl_module, "validation batch"
        )
        self.y_val.append(outputs["y"])
        self.y_hat_val.append(outputs["y_hat"])
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        y = torch.vstack([item for item in self.y_val])
        y_hat = torch.vstack([item for item in self.y_hat_val])
        loss = pl_module.loss_fn(y, y_hat)
        self.loss_dict["validation"][pl_module.current_epoch] = loss
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return super().on_validation_end(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,  #: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        performance_metrics = _log_performance_metrics(
            outputs, pl_module, "training batch"
        )
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return super().on_test_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,  #: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        return super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return super().on_test_end(trainer, pl_module)
