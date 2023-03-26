import argparse
import importlib

from pathlib import Path
from typing import Any, List

from clearml import Task

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from wawenet_trainer.callbacks import TestCallbacks
from wawenet_trainer.lightning_data import WEnetsDataModule
from wawenet_trainer.lightning_model import LitWAWEnetModule
from wawenet_trainer.training_config import clearml_config_exists, training_params
from wawenet_trainer.transforms import (
    AudioToTensor,
    get_normalizer_class,
    InvertAudioPhase,
    NormalizeAudio,
)

# main training script


def get_class(class_name: str, module_name: str) -> LitWAWEnetModule:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def setup_tb_logger(task_name: str, output_uri: str):
    tb_logger = TensorBoardLogger(output_uri, name=task_name)
    trainer_kwargs = {"logger": tb_logger}
    ptl_module_kwargs = {"task_name": task_name, "output_uri": output_uri}
    return trainer_kwargs, ptl_module_kwargs


def train(
    project_name: str = "",
    dataset_name: str = "",
    batch_size: int = None,
    training_epochs: int = None,
    seed: int = None,
    pred_metric: List[str] = "",
    groups_json: str = "",
    data_fraction: float = None,
    initial_weights: str = None,
    unfrozen_layers: int = None,
    segments: List[str] = "",
    csv_path: str = "",
    data_root_path: str = "",
    test_only: bool = False,
    lightning_module_name: str = "",
    initial_learning_rate: float = None,
    channels: int = None,
    num_workers: int = None,
    output_uri: Path = None,
    split_column_name: str = None,
    scatter_color_map: str = "Purples",
    logging: str = "local",
    **kwargs,
):
    target_list = "_".join(pred_metric)
    task_name = f"{target_list}_wawenets"

    # are we logging to clearML?
    if logging == "clearml":
        if clearml_config_exists():
            # clearML weirdly isn't finding stuff in `analysis.py`
            Task.add_requirements("matplotlib", "3.7.1")
            Task.add_requirements("pandas", "1.5.2")
            Task.add_requirements("scikit-learn", "1.2.2")
            Task.add_requirements("scikit-image", "0.19.3")
            task = Task.init(
                project_name=project_name,
                task_name=task_name,
                output_uri=str(
                    str(output_uri),
                ),
            )
            trainer_kwargs = {}
            ptl_module_kwargs = {
                "clearml_task": task,
                "task_name": task_name,
                "output_uri": output_uri,
            }
        else:
            print("falling back to local tensorboard logging")
            trainer_kwargs, ptl_module_kwargs = setup_tb_logger(task_name, output_uri)
    else:
        trainer_kwargs, ptl_module_kwargs = setup_tb_logger(task_name, output_uri)

    # TODO: correctly specify `worker_init_fn`
    # TODO: incorporate num workers
    pl.seed_everything(seed)

    normalizers = [
        get_normalizer_class(metric, norm_ind=ind)
        for ind, metric in enumerate(pred_metric)
    ]

    # set up the transforms/one augmentation
    speech_transforms = [
        NormalizeAudio(),
        AudioToTensor(),
    ]
    speech_transforms.extend(normalizers)

    augment_transforms = [
        NormalizeAudio(),
        InvertAudioPhase(),
        AudioToTensor(),
    ]
    augment_transforms.extend(normalizers)

    speech_transforms = transforms.Compose(speech_transforms)
    augment_transforms = transforms.Compose(augment_transforms)

    transform_list = [speech_transforms, augment_transforms]
    dataset_class = get_class(dataset_name, "wawenet_trainer.lightning_data")

    if not segments:
        segments = ["seg_1"]

    # next two items are a bit of a mess tbh
    fr_target_names = {
        "PESQMOSLQO",
        "POLQAMOSLQO",
        "PEMO",
        "ViSQOL3_c310",
        "STOI",
        "ESTOI",
        "SIIBGauss",
    }
    match_segments = [item for item in pred_metric if item in fr_target_names]

    # set up a data module to use with training
    data_module = WEnetsDataModule(
        csv_path,
        batch_size,
        data_root_path,
        pred_metric,
        transform_list,
        segments,
        dataset_class,
        subsample_percent=data_fraction,
        match_segments=match_segments,
        num_workers=num_workers,
        split_column_name=split_column_name,
    )
    data_module.setup()

    # grab the model class, IE LitWAWEnet2020
    # TODO: maybe this level of flexibility is unwarranted—it's a little confusing
    #       specifying the correct args
    lightning_module: LitWAWEnetModule = get_class(
        lightning_module_name, "wawenet_trainer.lightning_model"
    )
    model = lightning_module(
        initial_learning_rate,
        num_targets=len(pred_metric),
        channels=channels,
        unfrozen_layers=unfrozen_layers,
        weights_path=initial_weights,
        normalizers=normalizers,
        scatter_color_map=scatter_color_map,
        **ptl_module_kwargs,
    )

    # setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        TestCallbacks(normalizers=normalizers),
    ]

    # the progress bar is nice if training locally, but it clogs clearml logs
    progress_bar = True if logging == "local" else False

    # set up a trainer
    trainer = pl.Trainer(
        max_epochs=training_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        auto_select_gpus=True,
        auto_scale_batch_size="binsearch",
        enable_progress_bar=progress_bar,
        log_every_n_steps=10,  # TODO: relax this
        **trainer_kwargs,
    )

    if not test_only:
        trainer.fit(model=model, datamodule=data_module)

    test_output = trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="do a WAWEnets train/test cycle."
        "NOTE: defaults are defined in `../config/train_config.yaml`. any arguments "
        "not specified will revert to hopefully-sensical defaults."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="clearml project to be associated with task",
    )
    # TODO: remove references to old repo code
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of the dataset class to use from `wawenet_trainer.lightning_data`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for training and testing (default: 60)",
    )
    parser.add_argument(
        "--training_epochs",
        type=int,
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed pytorch and try for some consistent training (default: 1982)",
    )
    parser.add_argument(
        "--pred_metric",
        type=str,
        nargs="+",
        help="the metric to use as a target for training",
    )
    parser.add_argument(
        "--groups_json",
        type=str,
        help="file path pointing to a json containing group definitions",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        help="fraction of data to train with",
    )
    parser.add_argument(
        "--initial_weights",
        type=str,
        help="path to initial model weights",
    )
    parser.add_argument(
        "--unfrozen_layers",
        type=int,
        help="the number of param layers to leave unfrozen, "
        "counting back from the output of the model",
    )
    parser.add_argument(
        "--segments",
        type=str,
        nargs="+",
        help="names of the 3-second subsegments to include when training",
    )
    parser.add_argument(
        # TODO: remove references to old code
        "--csv_path",
        type=str,
        help="path pointing to the CSV that contains NISQA-style data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        help="path pointing to the root of the input audio files",
    )
    parser.add_argument(
        "--test_only",
        type=bool,
        help="if true, the net will not be trained before generating results.",
        default=False,
    )
    parser.add_argument(
        "--lightning_module_name",
        type=str,
        help="name of the model class that implements pl.LightningModule",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        help="learning rate starting place",
    )
    parser.add_argument(
        "--channels",
        type=int,
        help="number of convolutional channels to be used when training. default "
        "is 96",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="number of CPU workers that should be used to compute transforms/etc. "
        "and read data from disk",
    )
    parser.add_argument(
        "--output_uri",
        type=str,
        help="directory where experiment artifacts should be stored. defaults to "
        "`~/wenets_training_artifacts`",
    )
    parser.add_argument(
        "--training_regime",
        type=str,
        help="the specific type of model you'd like to train. options are `default`, "
        "`multitarget_obj_2022`, `multitarget_subj_obj_2022` and `multitarget_its_2022`.",
        default="default",
    )
    parser.add_argument(
        "--logging",
        type=str,
        help="how to track experiment results. options are `local` and `clearml`. the "
        "`clearml.conf` file must exist in the root of the home dir running the training "
        "script. if no conf file exists, the program will fall back to local tracking "
        "in an attempt to prevent inadvertent data leaks.",
        default="local",
    )

    args = parser.parse_args()
    # TODO: read configs for specific training regimes so you don't HAVE to deal with
    #       all these params

    # if args.training_regime is not specified, this will return a template dictionary
    train_params = training_params(args.training_regime)

    # override `train_params` with any arguments specified manually
    # this works because we defined the defaults in `../config/train_config.yaml`
    # instead of in the argparse arguments above
    manual_params = {key: val for key, val in vars(args).items() if val}
    train_params.update(**manual_params)

    train(**train_params)
