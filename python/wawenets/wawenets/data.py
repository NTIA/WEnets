import tempfile

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchaudio

from wawenets.generic_logger import construct_logger
from wawenets.stl_wrapper import LevelMeter, Resampler, SoxConverter, SpeechNormalizer


class RightPadSampleTensor:
    """
    zero-pad a segment to a specified `final_length`
    """

    # zero-pad a segment to a specified length

    def __init__(self, final_length):
        self.final_length = final_length

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        right pads or truncates the audio sample in `sample["sample_data"]` to
        `self.final_length`

        Parameters
        ----------
        sample : Dict[str, Any]
            a dictionary containing a key `sample_data` with a torch.Tensor
            value at minimum

        Returns
        -------
        Dict[str, Any]
            a dictionary containing `sample_data` that has been padded or
            truncated to `self.final_length`
        """
        # calculate how much to pad
        num_samples = sample["sample_data"].shape[1]
        pad_length = self.final_length - num_samples
        # unsqueeze and make a batch dim, i think this is the right place to do that
        sample["sample_data"] = sample["sample_data"].unsqueeze(0)
        if pad_length <= 0:
            return sample
        elif pad_length < 0:
            sample["sample_data"] = sample["sample_data"][:, : self.final_length]
            return sample
        padder = torch.nn.ConstantPad1d((0, pad_length), 0)
        # TODO: doublecheck below after all these changes
        sample["sample_data"] = padder(sample["sample_data"])
        return sample


class NormalizationCompensator:
    """there are small differences between the ITU normalization process and
    what is implemented in the MATLAB/C++ code. this attempts to compensate
    for those differences."""

    correction_factors = {
        48000: 8388608 / 8436450,
        32000: 8388608 / 8359694,
        24000: (8388608 / 8359694) * 8388608 / 8436450,
        16000: 1,
        8000: 8388608 / 8359694,
    }

    def __init__(self, input_sample_rate: int, perform_normalization: bool) -> None:
        self.factor = self.correction_factors[input_sample_rate]
        self.perform_normalization = perform_normalization

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        if requested by `self.perform_normalization`, performs additional
        normalization

        Parameters
        ----------
        sample : Dict[str, Any]
            a dictionary containing a key `sample_data` with a torch.Tensor
            value at minimum

        Returns
        -------
        Dict[str, Any]
            a dictionary containing `sample_data` that has been normalized by
            the tiniest amount
        """
        if self.perform_normalization:
            sample["sample_data"] = self.factor * sample["sample_data"]
        return sample


class WavHandler:
    """handles loading `.wav` files into a tensor suitable for input to WAWEnets

    right now, can only be used as a context manager"""

    def __init__(
        self,
        input_path: Path,
        level_normalization: bool,
        stl_bin_path: str,
        channel: int = 1,
    ) -> None:
        """
        init function for WavHandler class.

        Parameters
        ----------
        input_path : Path
            path to an input wav file
        level_normalization : bool
            whether or not to normalize input audio to -26 dBov
        stl_bin_path : str
            path to the directory containing compiled ITU STL binaries
        channel : int, optional
            the channel in the wav file to be used for processing, by default 1
        """
        self.input_path = input_path
        self.level_normalization = level_normalization
        self.stl_bin_path = Path(stl_bin_path)
        self.num_input_samples = 48000
        self.samples_per_second = 16000
        self.channel = channel
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
        self.downmixed_wav = None
        self.input_raw = None
        self.normalized_raw = None
        self.resampled_raw = None
        self.resampled_wav = None
        # wav file metadata
        self.metadata = None
        self.sample_rate = None
        self.duration = None
        self.segment_step_size = None
        self.resampled_frames = None
        # transforms
        self.compensator: NormalizationCompensator = None
        # loggggggg
        self.logger = construct_logger(self.__class__.__name__)

    def __enter__(self):
        """
        sets up infrastructure needed to prepare a wav file for inference

        Returns
        -------
        WavHandler
            WavHandler object used to preprocess wav inputs and postprocess
            inference results
        """
        # set up temp dir and intermediate file paths
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.downmixed_wav = self.temp_dir_path / "downmixed.wav"
        self.input_raw = self.temp_dir_path / "input.raw"
        self.normalized_raw = self.temp_dir_path / "normalized.raw"
        self.resampled_raw = self.temp_dir_path / "resampled.raw"
        self.resampled_wav = self.temp_dir_path / "resampled.wav"

        # store some metadata about our input file
        metadata = torchaudio.info(self.input_path)
        self.sample_rate = metadata.sample_rate
        self.duration = metadata.num_frames / self.sample_rate

        # warn about sample rate conversion if we need to
        self._warn_sample_rate()

        # grab the channel we're supposed to be working on
        self.converter.select_channel(self.input_path, self.downmixed_wav, self.channel)

        # convert to raw and resample since just about everything depends on that
        self.converter.wav_to_pcm(self.downmixed_wav, self.input_raw)
        self._resample()

        # convert to wav and gather some metadata, even though the resampled wav
        # won't be used after this. a little wasteful
        self.converter.pcm_to_wav(
            self.resampled_raw, self.resampled_wav, self.samples_per_second
        )
        metadata = torchaudio.info(self.resampled_wav)
        self.resampled_frames = metadata.num_frames

        # set up pytorch transform to do a little normalization compensation
        self.compensator = NormalizationCompensator(
            self.sample_rate, self.level_normalization
        )

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.temp_dir.cleanup()

    def _warn_sample_rate(self):
        """
        friendly warning that if you use these tools to resample your audio,
        accuracy will be affected slightly
        """
        if self.sample_rate != 16000:
            resample_warning = (
                f"native sample rate: {self.sample_rate}: "
                "when using the Python WAWEnet implementation to resample input data, "
                "accuracy decreases"
            )
            self.logger.warn(resample_warning)

    def _resample(self):
        """
        resamples an input raw file to `self.sample_rate`

        Raises
        ------
        RuntimeError
            if resampling fails
        """
        if not self.resampler.resample_raw(
            self.input_raw, self.resampled_raw, self.sample_rate
        ):
            raise RuntimeError(f"unable to resample {self.input_path}")

    def _normalize_raw(self, input_raw: Path, normalized_raw: Path) -> Path:
        """
        returns a path to the file that should be used after normalization

        Parameters
        ----------
        input_raw : Path
            path to the 16k samp/sec PCM input file
        normalized_raw : Path
            path to write the result of resampling if required.

        Returns
        -------
        Path
            the path to the resampled PCM file if resampling is required,
            otherwise, the path to the input PCM file.

        Raises
        ------
        RuntimeError
            if resampling fails.
        """
        # returns a path to the file that should be used after normalization
        if self.level_normalization:
            if not self.speech_normalizer.normalizer(input_raw, normalized_raw):
                raise RuntimeError(f"could not normalize {self.resampled_raw}")
        else:
            # if we've been instructed to not normalize, just use the resampled data
            normalized_raw = input_raw
        return normalized_raw

    def _load_wav(self, wav_path: Path) -> Tuple[torch.tensor, int]:
        """
        loads a wav file from disk and normalizes it to [-1, 1]

        Parameters
        ----------
        wav_path : Path
            path to a wav file

        Returns
        -------
        Tuple[torch.tensor, int]
            a tensor containing the normalized audio data, and an integer
            containing the sample rate of the file.
        """
        # load to tensor
        audio_data, sample_rate = torchaudio.load(wav_path)

        return audio_data, sample_rate

    def _calculate_pad_length(self, num_samples: int) -> int:
        """
        calculates the number of padding samples required to facilitate both
        an integer-number of 3-second segments and performing inference on
        all available data.

        Parameters
        ----------
        num_samples : int
            the number of samples available in an audio sample

        Returns
        -------
        int
            the number of samples the audio sample should be padded with
        """
        three_second_segs = num_samples // self.num_input_samples
        remainder = num_samples % self.num_input_samples
        if remainder:
            three_second_segs += 1
        return three_second_segs * self.num_input_samples

    def _calculate_num_segments(self, num_samples: int, stride: int) -> int:
        """
        calculates the number of segments to process given the total number of
        samples in the audio file and the requested stride

        Parameters
        ----------
        num_samples : int
            total number of samples available in the audio segment
        stride : int
            the stride by which neighboring predictions should be separated

        Returns
        -------
        int
            number of segments possible to process
        """
        return ((num_samples - self.num_input_samples) // stride) + 1

    def _calculate_start_stop_times(
        self, num_samples: int, stride: int
    ) -> List[Tuple[float, float, int]]:
        """
        generates a list of start and stop times based on the number of samples
        in the file and the specified stride

        Parameters
        ----------
        num_samples : int
            total number of samples available in the audio segment
        stride : int
            the stride by which neighboring predictions should be separated

        Returns
        -------
        Tuple[float, float, int]
            start time, stop time, and the segment number
        """
        start = 0
        start_stop_times = list()
        for seg_number in range(self._calculate_num_segments(num_samples, stride)):
            stop = start + self.num_input_samples
            start_stop_times.append(
                (
                    start / self.samples_per_second,
                    stop / self.samples_per_second,
                    seg_number,
                )
            )
            start += stride
        return start_stop_times

    def prepare_segment(
        self, start_time: float, end_time: float, seg_number: int
    ) -> Dict[str, Any]:
        """
        prepares a subsegment of an audio file specified by `start_time` and
        `end_time` for inference

        Parameters
        ----------
        start_time : float
            the time in seconds where the subsegment should begin
        end_time : float
            the time in seconds where the subsegment should end
        seg_number : int
            the segment number this start and end time represents

        Returns
        -------
        Dict[str, Any]
            a dictionary containing a key `sample_data` with a torch.Tensor
            value, and "active_level" and "speech_activity" keys
        """
        # we are using sox to trim because otherwise we'd be manually writing
        # samples to disk and doing a bunch of conversions in order to do the
        # measurements/etc.

        # calculate a path for a trimed PCM file
        trimmed_path = (
            self.resampled_raw.parent
            / f"{self.resampled_raw.stem}_seg_{seg_number}.raw"
        )
        # trim to start/end time
        self.converter.trim_pcm(self.resampled_raw, trimmed_path, start_time, end_time)

        # normalize segment by segment, just like in the MATLAB/C++ version
        normalized_raw = trimmed_path.parent / f"{trimmed_path.stem}_norm.raw"
        normalized_raw = self._normalize_raw(trimmed_path, normalized_raw)

        # convert the raw PCM to a wav file so torchaudio can read it
        normalized_wav = normalized_raw.parent / f"{normalized_raw.stem}.wav"
        self.converter.pcm_to_wav(normalized_raw, normalized_wav, 16000)

        # grab some measurements from the raw PCM file
        active_level, speech_activity = self.level_meter.measure(normalized_raw)

        # read the file and calculate how much padding is required
        sample, sample_rate = self._load_wav(normalized_wav)
        pad_length = self._calculate_pad_length(sample.shape[1])

        # set up a pytorch-style transform—figure out how to init this in __init__
        padder = RightPadSampleTensor(pad_length)

        # apply our transforms
        sample = {"sample_data": sample}
        sample = padder(sample)
        sample = self.compensator(sample)

        # send some metadata along for the ride
        sample["active_level"] = active_level
        sample["speech_activity"] = speech_activity

        return sample

    def prepare_tensor(self, stride: int) -> Tuple[torch.tensor, list, list, list]:
        """
        prepares a batch of tensors for inference and packages associated
        measurements for later use. this is where files that are longer
        than three seconds are handled.

        Parameters
        ----------
        stride : int
            the stride by which neighboring segments should be separated

        Returns
        -------
        Tuple[torch.tensor, list, list, list]
            a batch tensor containing waveform data from all segments, the
            measured active speech levels for each segment, the speech activity
            factor for each segment, and the start and stop times associated
            with each segment.
        """

        # for each valid segment, we have to:
        # 1. ask sox to write out the correct portion of a file to a new file
        # 2. normalize that file
        # 3. make measurements
        # 3. convert it to a wav
        # 4. read it
        # 5. pad the last segment if necessary
        # ----- above items can happen in `prepare_segment`
        # 6. pack the segments into a batch (!)

        # set up our storage data structures
        segments = list()
        active_levels = list()
        speech_activity = list()

        # calculate the start and stop times of all subsegments
        start_stop_times = self._calculate_start_stop_times(
            self.resampled_frames, stride
        )

        # read each subsegment and store measured attributes
        for seg in start_stop_times:
            segment = self.prepare_segment(*seg)
            segments.append(segment["sample_data"])
            active_levels.append(segment["active_level"])
            speech_activity.append(segment["speech_activity"])

        # create a batch so we can send all the segments to the CPU/GPU at once
        batch = torch.cat(segments)

        return batch, active_levels, speech_activity, start_stop_times

    def package_metadata(self) -> Dict[str, Any]:
        """
        packages metadata having to do with the input wavfile

        Returns
        -------
        Dict[str, Any]
            a dictionary containing information about the path of the input
            wavfile, the channel that was processed, the sample rate of the
            wavfile, the duration of the wav file, and whether normalization
            was performed.
        """
        return {
            "wavfile": self.input_path,
            "channel": self.channel,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "level_normalization": self.level_normalization,
        }
