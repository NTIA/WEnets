# python version

model files are plain `.pt` files, not traced JIT files, which should allow for better version compatibility. 
but the contents are the same as both the c++ model files and the onnx files.
the `.pt` model files in `./wawenets/weights` are plain pytorch model files, and are suitable for creating new traced JIT files for C++ or onnx in the future.

TODO: should probably specify a newer libtorch in C++ land.


CLI has these args:

# requirements
## sox
linux: 
```shell
apt install sox
```

macos:

the easiest way is to use [brew]() to install `sox`. install brew, then:

```shell
brew install sox
```
## itu-t stl 

this python code relies on [ITU-T STL](https://github.com/openitu/STL) executables in order to resample audio files and measure speech levels.
the tools request executable paths [via these mechanisms].

we're using this for some functions that are available in `torchaudio` because this way we can be certain that the inputs are the same among all different implementations (c++, matlab, etc.)

then do

```shell
cp config.yaml.template config.yaml
```

and point to the bin dir.

TODO: make an easy way to point to the STL tools in the python code. maybe i can write a shell script to clone the repo and put it into a specific place so the python tools know exactly where to find the executables.

## conda env

TODO: make an env.yaml

create the env:

```shell
conda create -n wenets python=3.10
conda activate wenets
pip install sox tqdm pyyaml
conda install pytorch torchvision torchaudio -c pytorch
```