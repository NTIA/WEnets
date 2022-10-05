# python version

the `.pt` model files in `./wawenets/weights` are plain pytorch model files, and are suitable for creating new traced JIT files for C++ or onnx in the future.

TODO: should probably specify a newer libtorch in C++ land.

CLI has these args:

# prerequisites
## sox
linux: 
```shell
apt install sox
```

macos:

the easiest way is to use [brew](https://brew.sh) to install `sox`. install brew, then:

```shell
brew install sox
```
## itu-t stl 

this python code relies on [ITU-T STL](https://github.com/openitu/STL) executables in order to resample audio files and measure speech levels.

we're using the STL utilities for some functions that are also available in `torchaudio` because this way we can be certain that the inputs are the same among all different implementations (c++, matlab, etc.)

firstly, clone the repo and compile the code using the instructions provided.

then, create a copy of `config.yaml.template` named `config.yaml`:

```shell
cp config.yaml.template config.yaml
```

and edit `config.yaml` to point to the `bin` dir where the STL tools have been compiled.

TODO: make an easy way to point to the STL tools in the python code. maybe i can write a shell script to clone the repo and put it into a specific place so the python tools know exactly where to find the executables.

## conda env

one way to install the python libraries required to run the python version of WENets is using anaconda (or miniconda). 
once anaconda is installed, use the following commands to set up and activate a new conda env:

```shell
conda env create -f wenets_env.yaml
conda activate wenets_dist
```

# running WENets
now, you should be able to run the following command and see its output:

```shell
python wawenets/wawenets.py --help
Usage: wawenets.py [OPTIONS]

  produces quality or intelligibility estimates for specified speech files.

Options:
  -m, --mode INTEGER     specifies a WAWEnet mode.
  -i, --infile TEXT      either a .wav file or a .txt file where each line
                         specifies a suitable .wavfile. if the latter, files
                         will be processed in sequence.  [required]
  -l, --level BOOLEAN    whether or not contents of a given .wav file should
                         be normalized. default is True.
  -s, --stride INTEGER   stride (in samples) on which to make predictions.
                         default is 48,000, meaningif a .wav file is longer
                         than 3 seconds, the model will generate a
                         predictionfor neighboring 3-second segments.
  -c, --channel INTEGER  specifies a channel to use if .wav file hasmultiple
                         channels. default is 1 using indices starting at 1
  -o, --output TEXT      path where a CSV file containing predictions should
                         be written. default isNone, and results are printed
                         to stdout
  --help                 Show this message and exit.
```
