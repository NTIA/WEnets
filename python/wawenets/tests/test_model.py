from pkg_resources import resource_filename

import pytest
import torch

from wawenets.model import WAWEnet2022


@pytest.fixture
def model_weights_2020():
    return resource_filename(
        "weights", "20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO_final_pytorch_eval.pt",
    )


def test_WAWEnet2022(model_weights_2020):
    script_module = torch.jit.load(model_weights_2020)
    model = WAWEnet2022()
    model.load_state_dict(script_module.state_dict())
