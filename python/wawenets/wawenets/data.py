import shutil
import tempfile

from pathlib import Path
from typing import Tuple

import torch
import torchaudio

from wawenets.stl_wrapper import Resampler, SoxConverter

# handle reading data/etc. here


class RightPadSampleTensor:
    """zero-pad a segment to a specified length"""

    def __init__(self, final_length):
        self.final_length = final_length

    def __call__(self, sample):
        # calculate how much to pad
        num_samples = sample["sample_data"].shape[0]
        pad_length = self.final_length - num_samples
        if pad_length <= 0:
            return sample
        elif pad_length < 0:
            sample["sample_data"] = sample["sample_data"][:, : self.final_length]
            return sample
        padder = torch.nn.ConstantPad1d((0, pad_length), 0)
        sample["sample_data"] = padder(sample["sample_data"])
        return sample


class WavHandler:
    """handles loading `.wav` files into a tensor suitable for input to WAWEnets

    right now, can only be used as a context manager"""

    def __init__(self, input_path: Path, stl_bin_path: str) -> None:
        self.input_path = input_path
        self.stl_bin_path = Path(stl_bin_path)
        self.path_to_filter = self.stl_bin_path / "filter"
        self.resampler = Resampler(self.path_to_filter)
        self.converter = SoxConverter()
        self.temp_dir = None
        self.temp_dir_path = None
        self.input_raw = None
        self.resampled_raw = None
        self.resampled_wav = None

    def __enter__(self):
        # set up temp dir and intermediate file paths
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.input_raw = self.temp_dir_path / "input.raw"
        self.resampled_raw = self.temp_dir_path / "resampled.raw"
        self.resampled_wav = self.temp_dir_path / "resampled.wav"

        # convert to raw and resample since just about everything depends on that
        self.converter.wav_to_pcm(self.input_path, self.input_raw)
        self.resample_wav()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.temp_dir.cleanup()

    def _copy_file(self, input_path: Path, output_path: Path):
        shutil.copy(input_path, output_path)
        return True

    def resample_raw(self, input_path: Path, output_path: Path, input_sample_rate: int):
        """resamples an input file to 16 kHz. returns true if successful."""
        resampler_map = {
            48000: self.resampler.down_48k_to_16k,
            32000: self.resampler.down_32k_to_16k,
            24000: self.resampler.down_24k_to_16k,
            16000: self._copy_file,  # a little wasteful
            8000: self.resampler.up_8k_to_16k,
        }

        return resampler_map[input_sample_rate](input_path, output_path)

    def resample_wav(self):
        metadata = torchaudio.info(self.input_path)
        if not self.resample_raw(
            self.input_raw, self.resampled_raw, metadata.sample_rate
        ):
            raise RuntimeError(f"unable to resample {self.input_path}")
        # convert to wav
        self.converter.pcm_to_wav(
            self.resampled_raw, self.resampled_wav, metadata.sample_rate
        )

    def load_wav(self, wav_path: Path) -> Tuple[torch.tensor, int]:
        # load to tensor
        audio_data, sample_rate = torchaudio.load(wav_path)

        return audio_data, sample_rate

    @staticmethod
    def calculate_pad_length(num_samples: int) -> int:
        """calculates the number of samples required to facilitate both
        an integer-number of 3-second segments and performing inference on
        all available data."""
        three_second_segs = num_samples // 48000
        remainder = num_samples % 48000
        if remainder:
            three_second_segs += 1
        return three_second_segs * 48000

    def prepare_tensor(self, channel: int = 1) -> torch.tensor:
        """channel specifies the channel to be used for inference; 1-based indexing"""

        # TODO: length, and stride, do the right things
        sample, sample_rate = self.load_wav(self.resampled_wav)
        pad_length = self.calculate_pad_length(sample.shape[1])
        padder = RightPadSampleTensor(pad_length)

        sample = {"sample_data": sample[channel - 1, :]}
        sample = padder(sample)

        return sample["sample_data"].unsqueeze(0).unsqueeze(0)
