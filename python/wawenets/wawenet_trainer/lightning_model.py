from pathlib import Path
from typing import Any

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from wawenets.model import WAWEnetICASSP2020, WAWEnet2020

# set up the lightning module here


class LitWAWEnetModule(pl.LightningModule):
    """parent module"""

    def __init__(
        self,
        learning_rate: float,
        *args: Any,
        weights_path: Path = None,
        unfrozen_layers: int = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()
        self.model = None
        self.unfrozen_layers = unfrozen_layers

        # we don't use this directly, but ptl does
        self.learning_rate = learning_rate
        self.example_input_array = torch.zeros((1, 1, 48000))

        # handle loading weights if we need to
        self.weights = None
        if weights_path:
            self.weights = torch.load(weights_path)

    def _freeze_layers(self):
        # freeze layers if requested
        if self.unfrozen_layers:
            all_params = list(self.model.parameters())
            for params in all_params[: -self.unfrozen_layers]:
                params.requires_grad = False

    def forward(self, batch):
        prediction = self.model(batch)
        return prediction

    def training_step(self, batch, batch_idx):
        # we will have a batch of audio tensors coming in and a batch of target
        # values as well
        x = batch["sample_data"]
        y = batch["pred_metric"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # now log to clearml
        # if we do `Task.init` correctly, this will send tensorboard-like logging
        # directly to clearML
        self.log("train_step_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        # `outputs` is a list of dictionaries for all batches in the epoch, returned
        # by `training_step`
        print(f"completed epoch {self.current_epoch}")
        train_loss_mean = torch.stack([item["loss"] for item in outputs]).mean()
        self.log("avg_train_loss", train_loss_mean)
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        x = batch["sample_data"]
        y = batch["pred_metric"]
        # we can keep track of a list of indices from the original DF here, even
        # though we probably don't need to since the val dataloader shouldn't have
        # a random batch order.
        df_ind = batch["df_ind"]
        y_hat = self.model(x)
        return {
            "val_step_loss": self.loss_fn(y_hat, y),
            "y": y,
            "y_hat": y,
            "df_ind": df_ind,
        }

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([item["val_step_loss"] for item in outputs]).mean()
        self.log("avg_val_loss", val_loss_mean)
        # TODO: not certain about this bit, the API might have changed
        super().validation_epoch_end(outputs)
        return {"avg_val_loss": val_loss_mean}

    def on_validation_end(self) -> None:
        """this will calculate average validation loss and log it"""
        # i think this is the right function for the new API?
        # actually i'm not certain it will get used.
        return super().on_validation_end()

    def test_step(self, batch, batch_idx, *args: Any, dataloader_idx=0, **kwargs: Any):
        x = batch["sample_data"]
        y = batch["pred_metric"]
        # we can keep track of a list of indices from the original DF here, even
        # though we probably don't need to since the val dataloader shouldn't have
        # a random batch order.
        df_ind = batch["df_ind"]
        y_hat = self.model(x)
        super().test_step(*args, **kwargs)
        # TDOO: hmm, maybe i should return a dict here instead of a tuple?
        return y, y_hat, df_ind

    def test_epoch_end(self, outputs) -> None:
        y_cumulative = list()
        y_hat_cumulative = list()
        # handle the case where we might have results from multiple dataloaders
        # if there are multiple dataloaders, results will be stored as separate
        # items in a list
        for loader_ind, loader_results in enumerate(outputs):
            # TODO: don't care about DF ind here?
            for batch_y, batch_y_hat, _ in loader_results:
                y_cumulative.append(batch_y)
                y_hat_cumulative.append(batch_y_hat)
        y = torch.hstack(y_hat_cumulative)
        y_hat = torch.vstack(y_hat_cumulative)
        loss = self.loss_fn(y_hat, y)
        self.log_dict({f"test_loader_{loader_ind}_loss": loss})
        return super().test_epoch_end(outputs)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler_kwargs = {
            "patience": 4,
            "verbose": True,
            "threshold": 1e-4,
            "cooldown": 1,
            "min_lr": 1e-8,
            "mode": "min",
        }
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_kwargs
            ),
            "monitor": "avg_val_loss",
        }
        # TODO: figure out how commenting this out affects the new API
        # return super().configure_optimizers()
        return [optimizer], [lr_scheduler]


class LitWAWEnetICASSP20202(LitWAWEnetModule):
    __version__ = "1.0.0"

    def __init__(
        self,
        learning_rate: float,
        *args: Any,
        num_targets: int = 1,
        channels: int = 96,
        **kwargs: Any,
    ) -> None:
        super().__init__(learning_rate, *args, **kwargs)

        # load the model
        self.model = WAWEnetICASSP2020(num_targets=num_targets, channels=channels)

        # if we've got some model weights
        if self.weights:
            self.model.load_state_dict(self.weights)

        # freeze some layers if you wanna do some transfer learning
        self._freeze_layers()


class LitWAWEnet2020(LitWAWEnetModule):
    __version__ = "1.0.0"

    def __init__(
        self,
        learning_rate: float,
        *args: Any,
        num_targets: int = 1,
        channels: int = 96,
        **kwargs: Any,
    ) -> None:
        super().__init__(learning_rate, *args, **kwargs)

        # load the model
        self.model = WAWEnet2020(num_targets=num_targets, channels=channels)

        # if we've got some model weights
        if self.weights:
            self.model.load_state_dict(self.weights)

        # freeze some layers if you wanna do some transfer learning
        self._freeze_layers()
