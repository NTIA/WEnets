from pkg_resources import resource_filename

import pytest
import torch

from wawenets.model import WAWEnet2020


@pytest.fixture
def model_weights_2020():
    weights_paths = [
        resource_filename("weights", "20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO.pt"),
        resource_filename("weights", "20200801_WAWEnetFD13FC96AvgReLU_POLQAMOSLQO_.pt"),
        resource_filename("weights", "20200802_WAWEnetFD13FC96AvgReLU_PEMO.pt"),
        resource_filename("weights", "20200802_WAWEnetFD13FC96AvgReLU_STOI.pt"),
    ]
    return weights_paths


def test_WAWEnet2020(model_weights_2020):
    for weight_path in model_weights_2020:
        weights = torch.load(weight_path)
        model = WAWEnet2020()
        model.load_state_dict(weights["model_state_dict"])
