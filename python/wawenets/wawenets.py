from pathlib import Path

import click

from wawenets import modeselektor
from wawenets.inference import Predictor

# do cli here

# TODO:
# 1. read CLI args
# 2. build a model
# 2. prepare data
# 3. call the model in a smart way
# 5. print out the results


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
    # set up our model
    config = modeselektor[mode]
    predictor = Predictor(**config)
    # make a prediction
    infile = Path(infile)
    prediction = predictor.predict(infile)
    print(prediction)


if __name__ == "__main__":
    cli()
