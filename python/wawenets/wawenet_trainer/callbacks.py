from typing import Any, List

import pytorch_lightning as pl

from wawenet_trainer.analysis import log_performance_metrics, WENetsAnalysis
from wawenet_trainer.lightning_model import LitWAWEnetModule

"""
set up callbacks here.

we have to do some gymnastics because:
1. `pl.Callback.on_[training/validation]_batch_end` receive outputs from the model
2. `pl.Callback.on_[training/validation]_epoch_end` don't receive outputs from the model
3. `pl.LightningModule.[training/validation]_epoch_end` in the pl.Module DO receive outputs from the
    model
"""


class WAWEnetCallbacks(pl.Callback):
    """a class to handle pytorch lightning callbacks; logs training/validation/test performance
    and initiates analysis where appropriate. register this class when instantiating pl.Trainer.

    unimplemented callbacks are omitted."""

    def __init__(self) -> None:
        self.epoch_losses = {
            "training": list(),
            "validation": list(),
        }
        self.epoch_correlations = {
            "training": list(),
            "validation": list(),
        }
        super().__init__()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: LitWAWEnetModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        uses `pl_module` to log loss and correlation.

        Parameters
        ----------
        trainer : pl.Trainer

        pl_module : LitWAWEnetModule

        outputs : dict
            delivered by LitWAWEnetModule.validation_step
        batch : Any
            contents of the current minibatch, both input and output.
        batch_idx : int
            batch index number, aka step number
        dataloader_idx : int
            index of the dataloader currently in use for training/validation
        """
        _ = log_performance_metrics(outputs, pl_module, "validation batch")
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: LitWAWEnetModule
    ) -> None:
        """
        uses data stored in `pl_module` to update the record of epoch losses and correlations

        Parameters
        ----------
        trainer : pl.Trainer

        pl_module : LitWAWEnetModule
        """
        # TODO: do i need the next two lines? i think this is a leftover from a less civilized time
        self.epoch_losses["validation"].append(pl_module.val_epoch_performance["loss"])
        self.epoch_correlations["validation"].append(
            pl_module.val_epoch_performance["correlations"]
        )
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: LitWAWEnetModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        uses `pl_module` to log loss and correlation.

        Parameters
        ----------
        trainer : pl.Trainer

        pl_module : LitWAWEnetModule

        outputs : dict
            delivered by LitWAWEnetModule.validation_step
        batch : Any
            contents of the current minibatch, both input and output.
        batch_idx : int
            batch index number, aka step number
        dataloader_idx : int
            index of the dataloader currently in use for training/validation
        """
        _ = log_performance_metrics(outputs, pl_module, "training batch")
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: LitWAWEnetModule
    ) -> None:
        """
        uses data stored in `pl_module` to log performance metrics and update
        the record of epoch losses and correlations

        Parameters
        ----------
        trainer : pl.Trainer

        pl_module : LitWAWEnetModule
        """
        performance_metrics = log_performance_metrics(
            pl_module.training_step_outputs, pl_module, "training epoch"
        )
        # TODO: do i need the next two lines? i think this is a leftover from a less civilized time
        self.epoch_losses["training"].append(performance_metrics["loss"])
        self.epoch_correlations["training"].append(performance_metrics["correlations"])
        pl_module.train_step_outputs = None
        return super().on_train_epoch_end(trainer, pl_module)

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: LitWAWEnetModule
    ) -> None:
        """
        here we do graph generation, denormalization, logging etc., for our test data
        and we do it for each test dataloader.

        Parameters
        ----------
        trainer : pl.Trainer

        pl_module : LitWAWEnetModule
        """
        # results from different test dataloaders will be in a list
        # here we do a bit of a hack to match them to the names assigned in
        # lightning_data.WEnetsDataModule.test_dataloader
        for dataloader_name, outputs in zip(
            trainer.datamodule.dataloader_names, pl_module.test_step_outputs
        ):
            analyzer = WENetsAnalysis(
                outputs, pl_module, dataloader_name=dataloader_name
            )
            analyzer.log_performance_metrics(dataloader_name)

            # log some stuff to clearml -- first, grouped performance based on impairment
            impairment_performance_df = analyzer.grouped_performance_metrics(
                "impairment"
            )
            pl_module.log_artifact(
                # bad name for backwards compatibility, i reserve the right to change it later
                f"{dataloader_name}_grouped_results_table_df",
                impairment_performance_df,
            )

            # now some per-condition measurements
            per_cond_df = analyzer.per_condition_metrics(impairment_performance_df)
            pl_module.log_artifact(f"{dataloader_name}_per_cond_df", per_cond_df)

            # now grouped performance based on language
            language_performanec_df = analyzer.grouped_performance_metrics("language")
            pl_module.log_artifact(
                f"{dataloader_name}_language_table_df", language_performanec_df
            )

            # overall results
            pl_module.log_artifact(
                f"{dataloader_name}_results_table_df",
                analyzer.target_performance_metrics(),
            )

            # kitchen sink—for the inevitable "what if we slice the data this
            # way?" question
            pl_module.log_artifact(f"{dataloader_name}_all_data_df", analyzer.df)

        # some stuff that i think we can get from scalars
        # TODO: can we get this data out of the clearML scalars?
        pl_module.log_artifact("training_corr", self.epoch_correlations["training"])
        pl_module.log_artifact("training_loss", self.epoch_losses["training"])
        pl_module.log_artifact("validation_corr", self.epoch_correlations["validation"])
        pl_module.log_artifact("validation_loss", self.epoch_losses["validation"])

        # clean out memory—perhaps not necessary
        pl_module.test_step_outputs = None
        return super().on_test_epoch_end(trainer, pl_module)
