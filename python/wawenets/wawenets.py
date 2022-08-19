import os
import yaml

from pathlib import Path

import click

from wawenets import modeselektor
from wawenets.data import WavHandler
from wawenets.inference import Predictor

# do cli here

# TODO:
# 1. read CLI args
# 2. build a model
# 2. prepare data
# 3. call the model in a smart way
# 5. print out the results


def export_results(results: list, out_file: str = None):
    """
    output line format:
    [wavfile] [channel] [sample_rate] [duration] [active_level] [speech_activity] [level_normalization] [segment_step_size] [WAWEnet_mode] [model_prediction]
    """
    pass


def get_stl_path():
    """returns the path to the STL bin dir based on the contents of
    config.yaml"""
    test_path = Path(os.path.realpath(__file__))
    config_path = test_path.parent.parent / "config.yaml"
    with open(config_path) as yaml_fp:
        config = yaml.safe_load(yaml_fp)
    return config["StlBinPath"]


def read_text_file(file_path: Path) -> list:
    """reads the lines from the given text file into a list"""
    with open(file_path) as fp:
        lines = fp.read().splitlines()
    return lines


@click.command()
@click.option(
    "-m",
    "--mode",
    help="specifies a WAWEnet mode.",
    required=False,
    type=click.INT,
    default="1",
)
@click.option(
    "-i",
    "--infile",
    help=(
        "either a .wav file or a .txt file where each line specifies a suitable .wav"
        "file. if the latter, files will be processed in sequence."
    ),
    type=click.STRING,
    required=True,
)
@click.option(
    "-l",
    "--level",
    help=(
        "whether or not contents of a given .wav file should be normalized. default "
        "is True."
    ),
    type=click.BOOL,
    required=False,
    default=True,
)
@click.option(
    "-s",
    "--stride",
    help=(
        "stride (in samples) on which to make predictions. default is 48,000, meaning"
        "if a .wav file is longer than 3 seconds, the model will generate a prediction"
        "for neighboring 3-second segments."
    ),
    type=click.INT,
    required=False,
    default=48000,
)
@click.option(
    "-c",
    "--channel",
    help=(
        "specifies a channel to use if .wav file has"
        "multiple channels. default is 1 using indices starting at 1"
    ),
    type=click.INT,
    required=False,
    default=1,
)
@click.option(
    "-o",
    "--output",
    help=("path where a text file containing predictions should be written"),
    type=click.STRING,
    required=False,
)
def cli(mode, infile, level, stride, channel, output):
    """produces quality or intelligibility estimates for specified speech
    files."""
    # read some config
    stl_path = get_stl_path()

    # set up our model
    config = modeselektor[mode]
    predictor = Predictor(**config)
    infile = Path(infile)

    # build up all the files that we need predictions for
    wav_files = list()
    if infile.suffix == ".wav":
        wav_files.append(infile)
    elif infile.suffix == ".txt":
        wav_paths = [Path(item) for item in read_text_file(infile)]
        wav_files.extend(wav_paths)

    # make prediction(s)
    for wav in wav_files:
        with WavHandler(wav, stl_path) as wh:
            prepared_tensor = wh.prepare_tensor(channel=channel)
            prediction = predictor.predict(prepared_tensor)

            print(prediction)


if __name__ == "__main__":
    cli()
