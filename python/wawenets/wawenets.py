import os
import yaml

from pathlib import Path
from typing import List

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

"""
file name
segment number
channel
sample rate
start time
stop time
active level
speech activity
level normalization
segment step size
mode
predictor 1 prediction
predictor 2 prediction
"""


def export_results(results: List[dict], out_file: Path = None):
    """
    either prints or writes to a file the results of processing.

    input is a list of dictionaries containing all relevant fields and optionally
    a file path specifying a location where results should be written.
    """
    line_format = (
        "{wavfile} {segment_number} {channel} {sample_rate} {start_time} {stop_time} "
        "{active_level} {speech_activity} {level_normalization} {segment_step_size} "
        "{WAWEnet_mode} {model_prediction}"
    )
    # generate the lines for each segment in each file
    lines = list()
    # loop over files
    for result in results:
        start_stop_times = result.pop("start_stop_times")
        active_levels = result.pop("active_levels")
        speech_activities = result.pop("speech_activities")
        per_seg_meta = zip(start_stop_times, active_levels, speech_activities)
        # loop over segments
        for segment_number, (
            (start_time, stop_time),
            active_level,
            speech_activity,
        ) in enumerate(per_seg_meta):
            sub_result = result.copy()
            sub_result.update(
                segment_number=segment_number,
                start_time=start_time,
                stop_time=stop_time,
                active_level=active_level,
                speech_activity=speech_activity,
            )
            lines.append(line_format.format(**sub_result))
    formatted = "\n".join(lines)
    if out_file:
        out_file.write_text(formatted)
    else:
        print(formatted)


def get_stl_path():
    """returns the path to the STL bin dir based on the contents of
    config.yaml"""
    current_path = Path(os.path.realpath(__file__))
    config_path = current_path.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"unable to find `config.yaml` in {config_path}. please follow the setup "
            "instructions in README.md to create `config.yaml"
        )
    with open(config_path) as yaml_fp:
        config = yaml.safe_load(yaml_fp)
    return config["StlBinPath"]


def read_text_file(file_path: Path) -> list:
    """reads the lines from the given text file into a list"""
    # hmmm, the text file should probably specify the correct channel to use
    # for each file too, 🤔
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
    help=(
        "path where a text file containing predictions should be written. default is"
        "None, and results are printed to stdout"
    ),
    type=click.STRING,
    required=False,
    default=None,
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
    if output:
        output = Path(output)

    # build up all the files that we need predictions for
    wav_files = list()
    if infile.suffix == ".wav":
        wav_files.append(infile)
    elif infile.suffix == ".txt":
        wav_paths = [Path(item) for item in read_text_file(infile)]
        wav_files.extend(wav_paths)

    # make prediction(s)
    predictions = list()
    for wav in wav_files:
        with WavHandler(wav, level, stl_path, channel=channel) as wh:
            (
                prepared_batch,
                active_levels,
                speech_activities,
                start_stop_times,
            ) = wh.prepare_tensor(stride)
            prediction = predictor.predict(prepared_batch)
            metadata = wh.package_metadata()
            metadata.update(
                segment_step_size=stride,
                WAWEnet_mode=mode,
                model_prediction=prediction,
                active_levels=active_levels,
                speech_activities=speech_activities,
                start_stop_times=start_stop_times,
            )
            predictions.append(metadata)

    export_results(predictions, output)


if __name__ == "__main__":
    cli()
