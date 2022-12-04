import argparse
import importlib

from clearml import Task

import pytorch_lightning as pl
from torchvision import transforms

from wawenet_trainer.lightning_data import TUBDataModule
from wawenet_trainer.transforms import AudioToTensor, InvertAudioPhase, NormalizeAudio

# main training script


def get_class(class_name, module_name):
    module = importlib.import_module(module_name)  # "nnate.models")
    return getattr(module, class_name)


def get_normalizer_class(target, bw="wb", norm_ind=0):
    module = importlib.import_module("nnate.nisqa_infra")
    prefix = "NormalizeNISQA"
    # TODO: make dictionary mapping
    class_name = f"{prefix}{target}"
    class_class = getattr(module, class_name)
    instance = class_class(norm_ind=norm_ind)
    return instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do a WAWEnets train/test cycle")
    parser.add_argument(
        "--project_name",
        type=str,
        help="clearml project to be associated with task",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="name of model to import from `nnate.models`",
    )
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
        help="names of the 3-second subsegments to include " "when training",
    )
    parser.add_argument(
        "--nisqa_csv_path",
        type=str,
        help="path pointing to the CSV that contains NISQA-" "style data",
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
        "--model_class_name",
        type=str,
        default="",
        help="the name of the model class that implements pl.LightningModule",
    )

    args = parser.parse_args()

    target_list = " ".join(args.pred_metric)
    # TODO: is this fixed yet?
    Task.add_requirements("setuptools", "59.5.0")
    task = Task.init(
        project_name=args.project_name,
        task_name="_".join(args.pred_metric) + "_from_script",
        output_uri="/home/whltexbread/ml_results",
    )

    normalizers = [
        get_normalizer_class(metric, norm_ind=ind)
        for ind, metric in enumerate(args.pred_metric)
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
    dataset_class = get_class(args.dataset_name, "wawenet_trainer.lightning_data")

    if args.segments:
        segments = args.segments
    else:
        segments = ["seg_1"]

    nisqa_csv_path = args.nisqa_csv_path

    fr_target_names = {
        "PESQMOSLQO",
        "POLQAMOSLQO",
        "PEMO",
        "ViSQOL3_C310",
        "STOI",
        "ESTOI",
        "SIIBGauss",
    }
    match_segments = [item for item in args.pred_metric if item in fr_target_names]
    data_module = TUBDataModule(
        nisqa_csv_path,
        60,
        "/ml_data/Speech/NISQA_Corpus_raw_16knorm",  # TODO: add to argparse
        args.pred_metric,
        transform_list,
        segments,
        subsample_percent=args.data_fraction,
        match_segments=match_segments,
    )
    data_module.setup()

    # here starts newer PTL interface, i think
    # grab the model class, IE LitWAWEnet2020
    model_class = get_class(args.model_class_name, "wawenet_trainer.lightning_model")
    # TODO: add learning rate and channels to argparse
    model = model_class(
        args.learning_rate, num_targets=len(args.pred_metric), channels=args.channels
    )

    # setup callbacks
    # nothing for now because callbacks aren't ready yet

    # set up a trainer
    trainer = pl.Trainer(
        max_epochs=args.training_epochs,
        callbacks=[],
        accelerator="auto",
        auto_select_gpus=True,
        auto_scale_batch_size="binsearch",
        enable_progress_bar=True,
    )

    trainer.fit(model=model, datamodule=data_module)
