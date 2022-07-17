import os

from pathlib import Path
from pkg_resources import resource_filename

import pytest
import torch

from wawenets.model import WAWEnet2020, WAWEnet2022


@pytest.fixture
def model_weights_2020():
    return resource_filename(
        "weights", "20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO_final_pytorch_eval.pt",
    )


def test_WAWEnet2020(model_weights_2020):
    print(os.getcwd())
    # TODO: things are named _really_ differently in the state dict of this JIT trace, i guess
    #       do another export without doing the JIT trace, things should line up?
    script_module = torch.jit.load(model_weights_2020)
    model = WAWEnet2022()
    model.load_state_dict(script_module.state_dict())
