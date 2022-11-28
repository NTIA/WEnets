# WAWEnets Python code

Implements Wideband Audio Waveform Evaluation networks or WAWEnets.

This WAWEnets implementation produces one or more speech quality or intelligibility values for each input speech signal without using reference speech signals.
WAWEnets are convolutional networks and they have been trained using full-reference objective speech quality and speech intelligibility values.

the `.pt` model files in `./wawenets/weights` are plain pytorch model files, and are suitable for creating new traced JIT files for C++ or ONNX in the future.

Details can be found in <a href="https://www.its.bldrdoc.gov/publications/3242.aspx" target="_blank">the ICASSP 2020 WAWEnets paper</a><sup id="wawenets">[1](#f1)</sup> and <a href="https://arxiv.org/abs/2206.13272" target="_blank">followup article</a><sup id="wawenets_article">[6](#f6)</sup>.

If you need to cite our work, please use the following:

```
@INPROCEEDINGS{
   9054204,
   author={A. A. {Catellier} and S. D. {Voran}},
   booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   title={Wawenets: A No-Reference Convolutional Waveform-Based Approach to Estimating Narrowband and Wideband Speech Quality},
   year={2020},
   volume={},
   number={},
   pages={331-335},
}
```

# Setup

In order to run the WAWEnets Python code, some initial setup is required.
Please follow the instructions below to prepare your machine and environment.

## SoX

[SoX](https://sox.sourceforge.net) is an audio processing library and CLI tool useful for format conversions, padding, and trimming, among other things.

To install SoX on a Debian-based Linux, use the `apt` package manager:

```shell
apt install sox
```

On macOS the easiest way to install SoX is by using [brew](https://brew.sh).
Follow the instructions to install brew, then use brew to install SoX:

```shell
brew install sox
```

In order to install SoX on Windows, follow the instructions on the [SoX SourceForge page](https://sox.sourceforge.net).

## ITU-T Software Tool Library (STL)

The Python WAWEnets implementation relies on [ITU-T STL](https://github.com/openitu/STL) executables in order to resample audio files and measure speech levels.
We're using the STL utilities for some functions that are also available in `torchaudio` because this allows us to be reasonably certain that the audio processing steps are the same among all WAWEnets implementations (C++, MATLAB, etc.)

First we must compile the STL executables.
To do this, clone the [STL repo](https://github.com/openitu/STL) and then follow the [build procedure](https://github.com/openitu/STL#build-procedure)

After the build procedure is complete, return to the WAWEnets Python implementation.
Create a copy of `config.yaml.template` named `config.yaml`:

```shell
cp config.yaml.template config.yaml
```

Edit `config.yaml` to point to the `bin` dir where the STL tools have been compiled, e.g. `/path/to/STL/bin`.

[//]: # (TODO: maybe i can write a shell script to clone the repo and put it into a specific place so the python tools know exactly where to find the executables.)

## Python Conda Environment

One way to install the Python libraries required to run the Python version of WAWENets is using [Anaconda](https://www.anaconda.com/products/distribution) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)). 
Once Anaconda or Miniconda are installed, use the following commands to set up and activate a new conda env:

```shell
conda env create -f wenets_env.yaml
conda activate wenets_dist
```

# Usage
After successfully completing the above steps, it should be possible to run the following command:

```shell
python wawenets/wawenets.py --help
```

and see its output:

```shell
Usage: wawenets.py [OPTIONS]

  the CLI interface Python WAWEnets. produces quality or intelligibility
  estimates for specified speech files.

Options:
  -m, --mode INTEGER     specifies a WAWEnet mode, default is 1
  -i, --infile TEXT      either a .wav file or a .txt file where each line
                         specifies a suitable .wav file. if the latter, files
                         will be processed in sequence.  [required]
  -l, --level BOOLEAN    whether or not contents of a given .wav file should
                         be normalized. default is True.
  -s, --stride INTEGER   stride (in samples @16k samp/sec) on which to make
                         predictions. default is 48,000, meaning if a .wav
                         file is longer than 3 seconds, the model will
                         generate a prediction for neighboring 3-second
                         segments.
  -c, --channel INTEGER  specifies a channel to use if .wav file has multiple
                         channels. default is 1 using indices starting at 1
  -o, --output TEXT      path where a CSV file containing predictions should
                         be written. default is None, and results are printed
                         to stdout
  --help                 Show this message and exit.
```

## Arguments

`infile` is either a .wav file or a .txt file where each line specifies a suitable .wav file. In this second case, the listed .wav files will be processed in sequence.

A suitable `.wav` file must:

- be uncompressed
- have sample rate 8, 16, 24, 32, or 48k smp/sec.
- contain at least 3 seconds of speech


To best match the designed scope of WAWEnets, the `.wav` file should have a speech activity factor of roughly 0.5 or greater and an active speech level near 26 dB below the clipping points of +/- 1.0 (see level normalization feature below). The native sample rate for WAWEnets is 16 k smp/sec so files with rates 8, 24, 32, or 48k rate are converted internally before processing.

`-m M` specifies a WAWEnet mode. The integer M specifies the WAWEnet trained using a specific full-reference target.

- `-m 1`: WAWEnet trained using WB-PESQ<sup id="wbpesq">[2](#f2)</sup> target values (Default)
- `-m 2`: WAWEnet trained using POLQA<sup id="polqa">[3](#f3)</sup> target values
- `-m 3`: WAWEnet trained using PEMO<sup id="pemo">[4](#f4)</sup> target values
- `-m 4`: WAWEnet trained using STOI<sup id="stoi">[5](#f5)</sup> target values
- `-m 5`: WAWEnet trained using seven objective targets: WB-PESQ, POLQA, STOI, PEMO, ViSQOL3 (c310), ESTOI, and SIIBGauss<sup>[6](#f6)</sup>
- `-m 6`: WAWEnet trained using four subjective targets (mos, noi, col, dis) and seven objective targets (WB-PESQ, POLQA, STOI, PEMO, ViSQOL3 (c310), ESTOI, and SIIBGauss)<sup>[6](#f6)</sup>

`-l L` specifies internal level normalization of `.wav` file contents to 26 dB below clipping.

- `-l 0`: normalization off
- `-l 1`: normalization on (Default)

`-s S` specifies specifies the segment step (stride) and is an integer with value 1 or greater.  Default is `-s 48,000`. WAWEnet requires a full 3 seconds of signal to generate a result.  If a `.wav` file is longer than 3 seconds multiple results may be produced. `S` specifies the number of samples to move ahead in the vector or file when extracting the next segment. The default value of 48,000 gives zero overlap between segments. Using this default any input less than 6 sec. will produce one result, based on just the first 3 sec. A 6 sec. input will produce two results. If `-s 24,000` for example, segment overlap will be 50%, a 4.5 sec. input will produce 2 results and a 6 sec. input will produce 3 results.

`-c C` specifies a channel number to use when the input speech is in a multi-channel `.wav` file. Default is `-c 1`.

`-o 'myFile.txt'` specifies a text file that captures WAWEnet results on a new line for each speech input processed. If the file exists it will be appended to. The extension `.txt` will be added as needed. Default is that no `.txt` file is generated.

## Outputs

The output for each of the N speech signals processed is in the format:

```
[row] [wavfile] [channel] [sample_rate] [duration] [level_normalization] [segment_step_size] [WAWEnet_mode] [segment_number] [start_time] [stop_time] [active_level] [speech_activity] [model_prediction]
```

where:

- `row` an identifier for the current row of output
- `wavfile` is the filename that has been processed
- `channel` is the channel of `wavfile` that has been processed
- `sample_rate` native sample rate of the `wavfile`
- `duration` duration of `wavfile` in seconds
- `level_normalization` reflects whether `wavfile` was normalized during processing
- `segment_step_size` reflects the segment step (stride) used to process `wavfile`
- `WAWEnet_mode` is the mode `wavfile` has been processed with
- `segment_number` is a zero-based index that indicates which segment of `wavfile` was processed
- `start_time` is the time in seconds where the current segment began within `wavfile`
- `stop_time` is the time in seconds where the current segment ended within `wavfile`
- `active_level` active speech level of the specified segment of `wavfile` in dB below overload
- `speech_activity` is the speech activity factor of the last specified segment of `wavfile`
- `model_prediction` output value produced by WAWEnet for the specified segment of `wavfile`

Internally, `pandas` is used to generate the text output.
If the `-o` option is specified, `pandas` generates a CSV and writes it to the given file path.

-------------------------------------
<b id="f1">1</b> Andrew A. Catellier & Stephen D. Voran, "WAWEnets: A No-Reference Convolutional Waveform-Based Approach to Estimating Narrowband and Wideband Speech Quality," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 331-335. [↩](#wawenets)

<b id="f2">2</b> ITU-T Recommendation P.862, "Perceptual evaluation of speech quality (PESQ)," Geneva, 2001.[↩](#wbqesq)

<b id="f3">3</b> ITU-T Recommendation P.863, "Perceptual objective listening quality analysis," Geneva, 2018.[↩](#polqa)

<b id="f4">4</b> R. Huber and B. Kollmeier, "PEMO-Q — A new method for objective audio quality assessment using a model of auditory perception," IEEE Trans. ASLP, vol. 14, no. 6, pp. 1902-1911, Nov. 2006.[↩](#pemo)

<b id="f5">5</b> C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," IEEE Trans. ASLP, vol. 19, no. 7, pp. 2125-2136, Sep. 2011.[↩](#stoi)

<b id="f6">6</b> Andrew Catellier & Stephen Voran, "Wideband Audio Waveform Evaluation Networks: Efficient, Accurate Estimation of Speech Qualities," arXiv preprint, Jun. 2022. [↩](#wawenets_article)