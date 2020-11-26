# ONNX weights

As stated on the [onnx.ai website](https://onnx.ai/):

> ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

If you'd like to include WAWEnets in your own product, the weights in this folder are easily convertible to formats used by other deep learning libraries.

Visit the [onnx.ai website](https://onnx.ai/) for more information.

## ONNX in Python

The following snippet demonstrates how one would load a model using Python and a generic ONNX runtime.

``` python 3
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import onnx
from onnxruntime import backend


def denormalize_target(
    normalized_target, target_min, target_max, input_min=-1, input_max=1
):
    """converts a normalized target to the equivalent value in
    [target_min, target_max]"""
    old_range = input_max - input_min
    target_range = target_max - target_min
    return \
        (((normalized_target - input_min) * target_range) / old_range) + \
        target_min


def load_wavs(paths):
    """loads each wav file in a list of paths and converts the
    data to the dtype and dimensions required  by the inference
    session"""
    wavs = []
    for path in paths:
        s_rate, speech = wavfile.read(path)
        speech = speech / 2 ** 16
        speech = speech.astype(np.float32)
        wavs.append(speech)
    return np.expand_dims(np.array(wavs), 1)


sample_paths = [
    Path("/home/Deep_Learner/data/Speech/401/T000053_Q159_D401.wav"),
    Path("/home/Deep_Learner/data/Speech/401/T000093_Q446_D401.wav"),
]

samples = load_wavs(sample_paths)

# min and max pesq values
min_val, max_val = 1.01, 4.64

# load the model weight and set up the inference session
weights_path = "/home/Deep_Learner/work/speech_learning/wb_results/20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO/20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO_final_onnx_eval.onnx"
onnx_model = onnx.load(weights_path)
inf_session = backend.prepare(onnx_model)

# run inference in a loop, ezpz
onnx_preds = []
for ind in range(samples.shape[0]):
    pred = backend.run(inf_session, np.expand_dims(samples[ind, :, :], 0))
    print(f"{sample_paths[ind].name}: {denormalize_target(pred[0].squeeze(), min_val, max_val)}")
```