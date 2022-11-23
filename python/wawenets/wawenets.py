"""
cli interface for Python WAWEnets implementation
"""

from pathlib import Path

import click

from wawenets import get_stl_path, modeselektor
from wawenets.data import WavHandler
from wawenets.inference import Predictor
from wawenets.postprocess import PostProcessor


def read_text_file(file_path: Path) -> list:
    """
    reads the lines from the given text file into a list

    Parameters
    ----------
    file_path : Path
        path to a text file

    Returns
    -------
    list
        contains the lines of the text file, one line per item
    """
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
        "stride (in samples @16k samp/sec) on which to make predictions. default is"
        "48,000, meaning if a .wav file is longer than 3 seconds, the model will"
        "generate a prediction for neighboring 3-second segments."
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
        "path where a CSV file containing predictions should be written. default is"
        "None, and results are printed to stdout"
    ),
    type=click.STRING,
    required=False,
    default=None,
)
def cli(
    mode: int = 1,
    infile: str = "",
    level: bool = True,
    stride: int = 48000,
    channel: int = 1,
    output: str = None,
):
    """
    the CLI interface for produces quality or intelligibility estimates for specified
    speech files.

    Parameters

    ----------

    mode : int, optional

        the WAWEnet mode that wil be used to process infile, by default 1

    infile : str, optional

        a path to either a wav file or a text file containing paths to wav files, by
        default ""

    level : bool, optional

        whether or not to normalize audio input before processing, by default True

    stride : int, optional

        stride (in samples @ 16k samp/sec) on which to make predictions, by default
        48000

    channel : int, optional

        the channel to make a prediction on if `infile` has more than one, by default 1

    output : str, optional

        path where a result should be written. if None, print to console, by default
        None
    """
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

    # format and output to selected location
    pp = PostProcessor(predictions)
    pp.export_results(output)


if __name__ == "__main__":
    cli()
