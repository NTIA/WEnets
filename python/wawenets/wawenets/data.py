from pathlib import Path

import torch
import torchaudio


# handle reading data/etc. here


class RightPadSampleTensor:
    """zero-pad a segment to a specified length"""

    def __init__(self, final_length):
        self.final_length = final_length

    def __call__(self, sample):
        # calculate how much to pad
        num_channels, num_samples = sample["sample_data"].shape
        pad_length = self.final_length - num_samples
        if pad_length <= 0:
            return sample
        elif pad_length < 0:
            sample["sample_data"] = sample["sample_data"][:, : self.final_length]
            return sample
        padder = torch.nn.ConstantPad1d((0, pad_length), 0)
        sample["sample_data"] = padder(sample["sample_data"])
        return sample


def load_audio_to_tensor(audio_path: Path) -> torch.tensor:
    padder = RightPadSampleTensor(48000)
    sample, sample_rate = torchaudio.load(audio_path)
    sample = {"sample_data": sample}
    sample = padder(sample)

    return sample["sample_data"].unsqueeze(0)
