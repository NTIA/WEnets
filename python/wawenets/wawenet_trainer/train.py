import argparse
import importlib

from typing import Any, List

from clearml import Task

import pytorch_lightning as pl
from torchvision import transforms

from wawenet_trainer.lightning_data import TUBDataModule
from wawenet_trainer.lightning_model import LitWAWEnetModule
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
):
    target_list = " ".join(pred_metric)
    # TODO: is this fixed yet?
    Task.add_requirements("setuptools", "59.5.0")
    task = Task.init(
        project_name=project_name,
        task_name=f"{target_list}_from_script",
        output_uri="/home/whltexbread/ml_results",
    )
    # TODO: we've gotta pass the task around so we can upload artifacts
    #       and some reporting too.

    # TODO: correctly specify `worker_init_fn`
    # TODO: incorporate num workers
    pl.seed_everything(seed)

    normalizers = [
        get_normalizer_class(metric, norm_ind=ind)
        for ind, metric in enumerate(pred_metric)
    ]

    # set up the dataset
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

    if segments:
        segments = segments
    else:
        segments = ["seg_1"]

    fr_target_names = {
        "PESQMOSLQO",
        "POLQAMOSLQO",
        "PEMO",
        "ViSQOL3_C310",
        "STOI",
        "ESTOI",
        "SIIBGauss",
    }
    match_segments = [item for item in pred_metric if item in fr_target_names]
    data_module = TUBDataModule(
        csv_path,
        batch_size,
        data_root_path,
        pred_metric,
        transform_list,
        segments,
        subsample_percent=data_fraction,
        match_segments=match_segments,
    )
    data_module.setup()

    # here starts newer PTL interface, i think
    # grab the model class, IE LitWAWEnet2020
    # TODO: maybe this level of flexibility is unwarranted—it's a little confusing
    #       specifying the correct args
    lightning_module = get_class(
        lightning_module_name, "wawenet_trainer.lightning_model"
    )
    model = lightning_module(
        initial_learning_rate,
        num_targets=len(pred_metric),
        channels=channels,
        unfrozen_layers=unfrozen_layers,
        weights_path=initial_weights,
    )

    # setup callbacks
    # nothing for now because callbacks aren't ready yet

    # set up a trainer
    trainer = pl.Trainer(
        max_epochs=training_epochs,
        callbacks=[],
        accelerator="auto",
        auto_select_gpus=True,
        auto_scale_batch_size="binsearch",
        enable_progress_bar=True,
    )

    if not test_only:
        trainer.fit(model=model, datamodule=data_module)

    test_output = trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do a WAWEnets train/test cycle")
    parser.add_argument(
        "--project_name",
        type=str,
        help="clearml project to be associated with task",
    )
    # TODO: remove references to old repo code
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="NISQADataset",
        help="name of the dataset class to import from `nnate.nisqa_infra`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=60,
        help="batch size for training and testing (default: 60)",
    )
    parser.add_argument(
        "--training_epochs",
        type=int,
        default=30,
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1982,
        help="seed pytorch and try for some consistent training",
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
        default=None,
        help="fraction of data to train with",
    )
    parser.add_argument(
        "--initial_weights",
        type=str,
        default=None,
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
        "--nisqa_csv_path",
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
        default="",
        help="name of the model class that implements pl.LightningModule",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=None,
        help="learning rate starting place",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=96,
        help="number of convolutional channels to be used when training. default "
        "is 96",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="number of CPU workers that should be used to compute transforms/etc. "
        "and read data from disk",
    )

    args = parser.parse_args()
    # TODO: read configs for specific training regimes so you don't HAVE to deal with
    #       all these params
