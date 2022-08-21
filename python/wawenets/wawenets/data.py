import shutil
import tempfile

from pathlib import Path
from typing import Tuple

import torch
import torchaudio

from wawenets.stl_wrapper import LevelMeter, Resampler, SoxConverter, SpeechNormalizer

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

    def __init__(
        self, input_path: Path, level_normalization: bool, stl_bin_path: str
    ) -> None:
        self.input_path = input_path
        self.level_normalization = level_normalization
        self.stl_bin_path = Path(stl_bin_path)
        # set up all our converters
        self.converter = SoxConverter()
        self.path_to_actlev = self.stl_bin_path / "actlev"
        self.level_meter = LevelMeter(self.path_to_actlev)
        self.path_to_filter = self.stl_bin_path / "filter"
        self.resampler = Resampler(self.path_to_filter)
        self.path_to_sv56 = self.stl_bin_path / "sv56demo"
        self.speech_normalizer = SpeechNormalizer(self.path_to_sv56)
        # set file paths
        self.temp_dir = None
        self.temp_dir_path = None
        self.input_raw = None
        self.normalized_raw = None
        self.resampled_raw = None
        self.resampled_wav = None
        # wav file metadata
        self.channel = None
        self.metadata = None
        self.sample_rate = None
        self.duration = None
        self.active_level = None
        self.speech_activity = None
        self.segment_step_size = None

    def __enter__(self):
        # set up temp dir and intermediate file paths
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.input_raw = self.temp_dir_path / "input.raw"
        self.normalized_raw = self.temp_dir_path / "normalized.raw"
        self.resampled_raw = self.temp_dir_path / "resampled.raw"
        self.resampled_wav = self.temp_dir_path / "resampled.wav"

        # store some metadata
        self.metadata = torchaudio.info(self.input_path)
        self.sample_rate = self.metadata.sample_rate
        self.duration = self.metadata.num_frames / self.sample_rate

        # convert to raw and resample since just about everything depends on that
        self.converter.wav_to_pcm(self.input_path, self.input_raw)
        self.resample()
        # normalize if requested
        self.normalize_raw()
        # convert to wav
        self.converter.pcm_to_wav(
            self.normalized_raw, self.resampled_wav, self.metadata.sample_rate
        )

        # now since we've got all the things, do a couple measurements
        self.active_level, self.speech_activity = self.level_meter.measure(
            self.normalized_raw
        )

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

    def resample(self):
        if not self.resample_raw(
            self.input_raw, self.resampled_raw, self.metadata.sample_rate
        ):
            raise RuntimeError(f"unable to resample {self.input_path}")

    def normalize_raw(self):
        if self.level_normalization:
            if not self.speech_normalizer.normalizer(
                self.resampled_raw, self.normalized_raw
            ):
                raise RuntimeError(f"could not normalize {self.resampled_raw}")
        else:
            # if we've been instructed to not normalize, just use the resampled data
            self.normalized_raw = self.resampled_raw

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

        # TODO: stride, do the right thing
        self.channel = channel
        sample, self.sample_rate = self.load_wav(self.resampled_wav)
        pad_length = self.calculate_pad_length(sample.shape[1])
        padder = RightPadSampleTensor(pad_length)

        sample = {"sample_data": sample[channel - 1, :]}
        sample = padder(sample)

        return sample["sample_data"].unsqueeze(0).unsqueeze(0)

    def package_metadata(self):
        return {
            "wavfile": self.input_path,
            "channel": self.channel,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "active_level": self.active_level,
            "speech_activity": self.speech_activity,
            "level_normalization": self.level_normalization,
        }
