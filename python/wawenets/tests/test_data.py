import os
import pytest
import yaml

from pathlib import Path
from pkg_resources import resource_filename

import torch


from wawenets.data import RightPadSampleTensor, WavHandler


@pytest.fixture
def test_wav():
    return resource_filename("speech", "T000053_Q159_D401_48.wav")


@pytest.fixture
def stl_path():
    test_path = Path(os.path.realpath(__file__))
    config_path = test_path.parent.parent / "config.yaml"
    with open(config_path) as yaml_fp:
        config = yaml.safe_load(yaml_fp)
    return config["StlBinPath"]


@pytest.fixture
def wav_handler(test_wav, stl_path):
    wh = WavHandler(test_wav, True, stl_path)
    return wh


def test_right_pad():
    padder = RightPadSampleTensor(48)
    # TODO: refactor to remove redundant code

    # smaller
    test_tensor = torch.zeros((24))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape[0] == 48

    # larger
    test_tensor = torch.zeros((49))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape[0] == 49

    # same size
    test_tensor = torch.zeros((48))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape[0] == 48


class TestWavHandler:
    def test_load_wav(self, wav_handler, test_wav):
        # this is a stupid test now
        result, sample_rate = wav_handler.load_wav(test_wav)
        assert result.shape == (1, 144000)
        assert result.dtype == torch.float32
        assert sample_rate == 48000

    def test_calculate_pad_length(self, wav_handler):
        assert wav_handler.calculate_pad_length(24000) == 48000
        assert wav_handler.calculate_pad_length(48001) == 96000

    def test_calculate_num_segments(self, wav_handler):
        assert wav_handler.calculate_num_segments(48000, 48000) == 1
        assert wav_handler.calculate_num_segments(48000, 24000) == 1
        assert wav_handler.calculate_num_segments(96000, 48000) == 2
        assert wav_handler.calculate_num_segments(96000, 24000) == 3
