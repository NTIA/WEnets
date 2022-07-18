# python version

TODO: maybe reexport the models using a newer version of pytorch? pytorch 1.12.0 will not read the models in this repo (created with torch 1.5.x). but of course we have to be careful not to break the c++ code—should probably specify a newer libtorch there as well.

uses plain torch, because

CLI has these args:

# conda env

TODO: make an env.yaml

create the env:

```shell
conda create -n wenets python=3.10
conda activate wenets
conda install pytorch torchvision torchaudio -c pytorch
```