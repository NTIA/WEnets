from typing import Any, List, Optional

import pytorch_lightning as pl

import torch

import numpy as np

from wawenet_trainer.analysis import log_performance_metrics, WENetsAnalysis
from wawenet_trainer.transforms import NormalizeGenericTarget

# set up callbacks here. this is the tricky one, maybe


# we have to do some stupid gymnastics because:
# 1. `pl.Callback.on_[training/validation]_batch_end` receive outputs from the model
# 2. `pl.Callback.on_[training/validation]_epoch_end` don't receive outputs from the model
# 3. `pl.LightningModule.[training/validation]_epoch_end` in the pl.Module DO receive outputs from the
#     model
#
# otherwise we'd have some nice separation


class TestCallbacks(pl.Callback):
    def __init__(self, normalizers: List[NormalizeGenericTarget] = None) -> None:
        self.normalizers = normalizers
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
        performance_metrics = log_performance_metrics(
            outputs, pl_module, "validation batch"
        )
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        performance_metrics = log_performance_metrics(
            pl_module.val_step_outputs, pl_module, "validation epoch"
        )
        # TODO: do i need the next line? i think this is a leftover from a less civilized time
        self.loss_dict["validation"].append(performance_metrics["loss"])
        pl_module.val_loss_mean = performance_metrics["loss"]
        pl_module.val_step_outputs = None
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
        performance_metrics = log_performance_metrics(
            outputs, pl_module, "training batch"
        )
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        performance_metrics = log_performance_metrics(
            pl_module.training_step_outputs, pl_module, "training epoch"
        )
        # TODO: do i need the next line? i think this is a leftover from a less civilized time
        self.loss_dict["training"].append(performance_metrics["loss"])
        pl_module.train_step_outputs = None
        return super().on_train_epoch_end(trainer, pl_module)

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

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # TODO: handle denormalization here, since this is where we're doing reporting
        # TODO: handle reporting by language, impairment here
        # lots of work to do here—graph generation, denormalization, etc.
        analyzer = WENetsAnalysis(pl_module.test_step_outputs, pl_module)
        analyzer.log_performance_metrics()
        analyzer.grouped_performance_metrics("impairment")
        pl_module.test_step_outputs = None
        return super().on_test_epoch_end(trainer, pl_module)

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return super().on_test_end(trainer, pl_module)
