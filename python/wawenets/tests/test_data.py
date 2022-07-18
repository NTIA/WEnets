from pkg_resources import resource_filename

import pytest

import torch


from wawenets.data import RightPadSampleTensor, load_audio_to_tensor


def test_right_pad():
    padder = RightPadSampleTensor(48)
    # TODO: refactor to remove redundant code

    # smaller
    test_tensor = torch.zeros((1, 24))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape == (1, 48)

    # larger
    test_tensor = torch.zeros((1, 49))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape == (1, 49)

    # same size
    test_tensor = torch.zeros((1, 48))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape == (1, 48)


@pytest.fixture
def test_wav():
    return resource_filename("speech", "T000053_Q159_D401_48.wav")


def test_load_audio_to_tensor(test_wav):
    result = load_audio_to_tensor(test_wav)
    assert result.shape == (1, 1, 144000)
    assert result.dtype == torch.float32