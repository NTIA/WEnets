import pytest

import torch
import torchaudio

from wawenets import get_stl_path
from wawenets.data import RightPadSampleTensor, WavHandler


@pytest.fixture
def test_wav(tmp_path):
    wav_file = tmp_path / "wav_file.wav"
    sample_rate = 48000
    tensor_length = 3 * sample_rate
    noise_tensor = torch.rand((1, tensor_length))
    torchaudio.save(wav_file, noise_tensor, sample_rate)
    return wav_file


@pytest.fixture
def stl_path():
    return get_stl_path()


@pytest.fixture
def wav_handler(test_wav, stl_path):
    wh = WavHandler(test_wav, True, stl_path)
    return wh


def test_right_pad():
    padder = RightPadSampleTensor(48)

    # smaller
    test_tensor = torch.zeros((1, 24))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape[2] == 48

    # larger
    test_tensor = torch.zeros((1, 49))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape[2] == 49

    # same size
    test_tensor = torch.zeros((1, 48))
    result = padder({"sample_data": test_tensor})
    assert result["sample_data"].shape[2] == 48


class TestWavHandler:
    def test_load_wav(self, wav_handler, test_wav):
        # i guess this really only tests that torchaudio is installed properly
        result, sample_rate = wav_handler._load_wav(test_wav)
        assert result.shape == (1, 144000)
        assert result.dtype == torch.float32
        assert sample_rate == 48000

    def test_calculate_pad_length(self, wav_handler):
        assert wav_handler._calculate_pad_length(24000) == 48000
        assert wav_handler._calculate_pad_length(48001) == 96000

    def test_calculate_num_segments(self, wav_handler):
        assert wav_handler._calculate_num_segments(48000, 48000) == 1
        assert wav_handler._calculate_num_segments(48000, 24000) == 1
        assert wav_handler._calculate_num_segments(72000, 24000) == 2
        assert wav_handler._calculate_num_segments(96000, 48000) == 2
        assert wav_handler._calculate_num_segments(96000, 24000) == 3

    def test_calculate_start_stop_times(self, wav_handler):
        # one 3-sec file, stride 48000
        expected = [(0, 3, 0)]
        assert wav_handler._calculate_start_stop_times(48000, 48000) == expected
        # one 3-sec file, stride 24000
        expected = [(0, 3, 0)]
        assert wav_handler._calculate_start_stop_times(48000, 24000) == expected
        # one 4.5-sec file, stride 24000
        expected = [(0, 3, 0), (1.5, 4.5, 1)]
        assert wav_handler._calculate_start_stop_times(72000, 24000) == expected
        # one 6-sec file, stride 48000
        expected = [(0, 3, 0), (3, 6, 1)]
        assert wav_handler._calculate_start_stop_times(96000, 48000) == expected
        # one 6-sec file, stride 24000
        expected = [(0, 3, 0), (1.5, 4.5, 1), (3, 6, 2)]
        assert wav_handler._calculate_start_stop_times(96000, 24000) == expected
