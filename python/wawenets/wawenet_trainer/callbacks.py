from typing import Any, Optional

import pytorch_lightning as pl

# set up callbacks here. this is the tricky one, maybe


class TestCallbacks(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self._y = None
        self._y_hat = None

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._y, self._y_hat = list(), list()
        return super().on_validation_start(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._y.extend(outputs["y"].toList())
        self._y_hat.extend(outputs["y_hat"].tolist())
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return super().on_validation_end(trainer, pl_module)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return super().on_test_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
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
