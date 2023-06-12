import json

from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any, List, Union

import torch
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl

from clearml import Task

from wawenets.model import WAWEnetICASSP2020, WAWEnet2020
from wawenet_trainer.log_performance import log_performance_metrics
from wawenet_trainer.transforms import NormalizeGenericTarget

# set up the lightning module here


class LitWAWEnetModule(pl.LightningModule):
    """implements main training logic for WAWEnets training. child classes
    should implement specific models and load weights if requested."""

    def __init__(
        self,
        learning_rate: float,
        *args: Any,
        weights_path: Path = None,
        unfrozen_layers: int = None,
        normalizers: List[NormalizeGenericTarget] = None,
        clearml_task: Task = None,
        task_name: str = "default_task_name",
        scatter_color_map: str = None,
        output_uri: str = None,
        init_timestamp: str = "",
        **kwargs: Any,
    ) -> None:
        """
        initializes LitWAWEnetModule

        Parameters
        ----------
        learning_rate : float
            the learning rate to be used when training begins
        weights_path : Path, optional
            where pretrained model weights are located, for use when
            attempting transfer learning, by default None
        unfrozen_layers : int, optional
            the number of layers to train, starting from model output and
            working backwards. in this case "layers" means individual network
            components, not sections of the WAWEnet.
        normalizers : List[NormalizeGenericTarget], optional
            contains target-specific normalizers in the order they will be
            delivered by `WEnetsDataModule`. these are primarily used by
            analysis callbacks. TODO: this should not be optional, by default None
        clearml_task : Task, optional
            the clearML task that should be used for logging, by default None
        task_name : str, optional
            name used for storing results on disk and for logging to clearML,
            by default "default_task_name"
        scatter_color_map : str, optional
            name of the matplotlib colormap that should be used when generating
            result 2D histograms, by default None
        output_uri : str, optional
            directory where experiment artifacts should be stored, defaults to
            `~/wenets_training_artifacts`
        init_timestamp : str, optional
            a timestamp that will be prepended to the experiment directory
            when logging locally, defaults to a modified ISO format marked
            at the begining of the training process
        """
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()
        self.model: torch.nn.Module = None
        self.unfrozen_layers = unfrozen_layers
        self.normalizers = normalizers
        self.clearml_task = clearml_task
        self.task_name = task_name
        self.output_uri = output_uri
        self.scatter_color_map = scatter_color_map

        # if there's no init time, make one. this helps us keep track of training artifacts
        # in the absense of an experiment tracking system
        self.init_timestamp = init_timestamp
        if not self.init_timestamp:
            now = datetime.now().replace(microsecond=0).isoformat()
            self.init_timestamp = now.replace(":", "-")

        # we don't use this directly, but ptl does
        self.learning_rate = learning_rate
        self.example_input_array = torch.zeros((1, 1, 48000))

        # handle loading weights if we need to
        self.weights = None
        if weights_path:
            self.weights = torch.load(weights_path)

        # some structured storage for train/val/test analysis
        self.training_step_outputs: list = None
        self.val_epoch_performance: dict = None
        self.test_step_outputs: list = None

    def _freeze_layers(self):
        """
        freeze layers if requested by `self.unfrozen_layers`.
        """
        # freeze layers if requested
        if self.unfrozen_layers:
            all_params = list(self.model.parameters())
            for params in all_params[: -self.unfrozen_layers]:
                params.requires_grad = False

    def _log_artifact_to_disk(
        self, artifact_name: str, artifact: Union[pd.DataFrame, dict, list, plt.Figure]
    ):
        """
        implements rudimentary logging; writes dataframes to disk as json and
        converts other objects to json before writing to disk

        Parameters
        ----------
        artifact_name : str
            the name of the artifact that should be logged. if the artifact is a
            pandas dataframe and the artifact name does not include "df", `_df`
            will be appended.
        artifact : Union[pd.DataFrame, dict, list]
            the artifact that should be logged.

        Raises
        ------
        RuntimeError
            raised if a logging method for the artifact's type is not implemented.
        """
        # i guess we assume here that `self.output_uri` is a local path
        output_path = Path(self.output_uri) / self.task_name / self.init_timestamp
        output_path.mkdir(exist_ok=True)
        artifact_path = output_path / f"{artifact_name}.json"
        if isinstance(artifact, pd.DataFrame):
            # ensure there's something in the filename that lets future you know
            # the contents are suitable for loading into a pandas dataframe
            if "df" not in artifact_name:
                artifact_path = output_path / f"{artifact_name}_df.json"
            artifact.to_json(artifact_path)
        elif isinstance(artifact, dict) or isinstance(artifact, list):
            # kind of ugly way to detect and handle tensors before serializing
            if isinstance(artifact, list) and isinstance(artifact[0], torch.Tensor):
                artifact = [float(item) for item in artifact]
            with open(artifact_path, "w") as json_fp:
                json.dump(artifact, json_fp, default=str)
        elif isinstance(artifact, plt.Figure):
            # all figures are pdfs because all conference papers are LaTeX docs
            artifact_path = artifact_path.parent / f"{artifact_path.stem}.pdf"
            artifact.savefig(artifact_path, bbox_inches="tight")
        else:
            raise RuntimeError("Is your artifact generator running? better go catch it")

    def log_artifact(self, artifact_name: str, artifact: Any):
        """
        logs artifacts to the correct place, whether clearML or locally

        Parameters
        ----------
        artifact_name : str
            the name of the artifact that will be logged
        artifact : Any
            the artifact that should be logged.
        """
        if not self.clearml_task:
            self._log_artifact_to_disk(artifact_name, artifact)
            return
        self.clearml_task.upload_artifact(artifact_name, artifact)

    def log_figure(self, title: str, series: str, figure: plt.Figure):
        """
        logs figures to the correct place, whether clearML or locally

        Parameters
        ----------
        title : str
            title of the plot, usually based on the target name
        series : str
            series associated with the plot, usually based on the dataloader name
        figure : plt.Figure
            figure to be stored/logged
        """
        if not self.clearml_task:
            figure_name = "_".join([series, title])
            self._log_artifact_to_disk(figure_name, figure)
            return
        # make the figure show up in the clearML "plots" tab
        self.clearml_task.logger.report_matplotlib_figure(
            title=title, series=series, figure=figure, iteration=self.global_step
        )
        # make the figure show up in the clearML "debug samples" tab, because
        # from there you can actually download a version of the fig that works
        with BytesIO() as buffer:  # til `BytesIO` has a context manager!
            # all figures are pdfs because all conference papers are LaTeX docs
            # TODO: does `report_media` make it possible to get the PDFs via the API?
            figure.savefig(buffer, format="pdf", bbox_inches="tight")
            self.clearml_task.logger.report_media(
                f"{series}_{title}",
                series=series,
                stream=buffer,
                file_extension="pdf",
                iteration=self.global_step,
            )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        forward pass on the model.

        Parameters
        ----------
        batch : dict
            generated by the `__call__` methods in WAWEnet-style datasets

        Returns
        -------
        torch.Tensor
            predictions generated by the model
        """
        prediction = self.model(batch)
        return prediction

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        performs a training step on one minibatch.

        Parameters
        ----------
        batch : dict
            generated by the `__call__` methods in WAWEnet-style datasets
        batch_idx : int
            step number of the current minibatch

        Returns
        -------
        dict
            contains the loss calculated over this minibatch, normalized target
            values, and normalized model predictions
        """
        # we will have a batch of audio tensors coming in and a batch of target
        # values as well
        x = batch["sample_data"]
        y = batch["pred_metric"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # now log to clearml
        # if we do `Task.init` correctly, this will send tensorboard-like logging
        # directly to clearML
        self.log("training batch loss", loss)
        return {
            "loss": loss,
            "y": y.detach().cpu(),
            "y_hat": y_hat.detach().cpu(),
        }

    def training_epoch_end(self, outputs: List[dict]):
        """
        housekeeping performed at the end of a training epoch

        Parameters
        ----------
        outputs : List[dict]
            a list of dictionaries for all batches in the epoch, returned
            by `self.training_step`
        """
        print(f"completed epoch {self.current_epoch}")
        # this is the sanctioned way to get the outputs to the callback/somewhere
        # other than this method:
        # https://lightning.ai/docs/pytorch/stable/common/
        #   lightning_module.html#train-epoch-level-operations
        self.training_step_outputs = outputs
        return super().training_epoch_end(outputs)

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        performs a validation step on one minibatch.

        Parameters
        ----------
        batch : dict
            generated by the `__call__` methods in WAWEnet-style datasets
        batch_idx : int
            step number of the current minibatch

        Returns
        -------
        dict
            contains the loss calculated over this minibatch, normalized target
            values, and normalized model predictions
        """
        x = batch["sample_data"]
        y = batch["pred_metric"]
        y_hat = self.model(x)
        step_loss = self.loss_fn(y_hat, y)
        self.log("validation batch loss", step_loss)
        return {
            "val_batch_loss": step_loss,
            "y": y.detach().cpu(),
            "y_hat": y_hat.detach().cpu(),
        }

    def validation_epoch_end(self, outputs: List[dict]):
        """
        housekeeping performed at the end of a validation epoch

        Parameters
        ----------
        outputs : List[dict]
            a list of dictionaries for all batches in the epoch, returned
            by `self.validation_step`
        """
        # doing loss/correlation calculations here instead of in the callback because we need
        # the loss for the learning rate scheduler and there's not a way that i know of to get
        # data from a callback back to here.
        performance_metrics = log_performance_metrics(outputs, self, "validation epoch")
        self.val_epoch_performance = performance_metrics

        return super().validation_epoch_end(outputs)

    def test_step(
        self,
        batch: dict,
        batch_idx: int,
        *args: Any,
        dataloader_idx: int = 0,
        **kwargs: Any,
    ) -> dict:
        """
        performs a test step on one minibatch.

        Parameters
        ----------
        batch : dict
            generated by the `__call__` methods in WAWEnet-style datasets
        batch_idx : int
            step number of the current minibatch
        dataloader_idx : int, optional
            the index of the dataloader currently being used to generate
            minibatches, by default 0

        Returns
        -------
        dict
            contains the loss calculated over this minibatch, normalized target
            values, and normalized model predictions
        """
        x = batch["sample_data"]
        y = batch["pred_metric"]

        # keep track of row metadata so we don't have to back it out later
        language = batch["language"]
        impairment = batch["impairment"]
        # predict
        y_hat = self.model(x)
        super().test_step(*args, **kwargs)
        return {
            "test_step_loss": self.loss_fn(y_hat, y).detach().cpu(),
            "y": y.detach().cpu(),
            "y_hat": y_hat.detach().cpu(),
            "language": language,
            "impairment": impairment,
        }

    def test_epoch_end(self, outputs: Union[List[List[dict]], List[dict]]) -> None:
        """
        housekeeping performed at the end of a test epoch

        Parameters
        ----------
        outputs : List[List[dict]]
            a list of dictionaries for all batches in the epoch, returned
            by `self.test_step`
        """
        # the shenanigans below are to handle inconsistency with how PTL delivers outputs
        # from the test step when you have one or many test dataloaders.
        #
        # i'm probably not understanding something about the API, but this works
        if isinstance(outputs[0], dict):
            self.test_step_outputs = [outputs]
        elif isinstance(outputs[0], list):
            self.test_step_outputs = outputs
        return super().test_epoch_end(outputs)

    def configure_optimizers(self) -> Any:
        """
        sets up the Adam scheduler and the ReduceLROnPlateau learning rate
        scheduler.

        Returns
        -------
        Any
            a tuple of lists, containing the optimizer and the learning rate
            scheduler, respectively. lightning allows for different optimizers
            and learning rate schedulers for different parts of the model and
            we are not taking advantage of that here.
        """
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
            # the key for this monitor has to be `self.log` somewhere
            "monitor": "validation epoch loss",
        }
        return [optimizer], [lr_scheduler]


class LitWAWEnetICASSP20202(LitWAWEnetModule):
    """
    implements a LitWAWEnetModule where the model described in the 2020 ICASSP
    paper is trained
    """

    __version__ = "1.0.0"

    def __init__(
        self,
        learning_rate: float,
        *args: Any,
        num_targets: int = 1,
        channels: int = 96,
        **kwargs: Any,
    ) -> None:
        """
        initializes LitWAWEnetICASSP20202

        Parameters
        ----------
        learning_rate : float
            the learning rate to be used when training begins
        num_targets : int, optional
            the number of targets this model should be trained to predict, by default 1
        channels : int, optional
            number of convolutional channels used to generate features, by default 96
        """
        super().__init__(learning_rate, *args, **kwargs)

        # load the model
        self.model = WAWEnetICASSP2020(num_targets=num_targets, channels=channels)

        # if we've got some model weights
        if self.weights:
            self.model.load_state_dict(self.weights)

        # freeze some layers if you wanna do some transfer learning
        self._freeze_layers()


class LitWAWEnet2020(LitWAWEnetModule):
    """
    implements a LitWAWEnetModule where the model described in the 2023 arxiv
    paper is trained
    """

    __version__ = "1.0.0"

    def __init__(
        self,
        learning_rate: float,
        *args: Any,
        num_targets: int = 1,
        channels: int = 96,
        **kwargs: Any,
    ) -> None:
        """
        initializes LitWAWEnet

        Parameters
        ----------
        learning_rate : float
            the learning rate to be used when training begins
        num_targets : int, optional
            the number of targets this model should be trained to predict, by default 1
        channels : int, optional
            number of convolutional channels used to generate features, by default 96
        """
        super().__init__(learning_rate, *args, **kwargs)

        # load the model
        self.model = WAWEnet2020(num_targets=num_targets, channels=channels)

        # if we've got some model weights
        if self.weights:
            self.model.load_state_dict(self.weights)

        # freeze some layers if you wanna do some transfer learning
        self._freeze_layers()
