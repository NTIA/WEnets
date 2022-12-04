import shutil
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import Any, Dict, Tuple

import click
import sox

import numpy as np

from wawenets import get_stl_path
from wawenets.generic_logger import construct_logger


class SoxConverter:
    """
    wrapper around the pysox API that performs some specific operations that
    are useful to WAWEnet operation

    pysox: https://github.com/rabitt/pysox
    """

    wav = ".wav"
    raw = ".raw"

    def __init__(self):
        # loggggggg
        self.logger = construct_logger(self.__class__.__name__)

    def _convert(self, input_path: Path, output_path: Path, sample_rate: int = None):
        """
        converts from wav to raw (pcm) or pcm (raw) to wav depending on the
        input arguments

        Parameters
        ----------
        input_path : Path
            the file path to the file that will be converted.
        output_path : Path
            the file path where tho converted file should be written
        sample_rate : int, optional
            the sample rate of the associated input file if the input file
            is not a .wav file, by default None
        """
        # assumes input path is a one-channel audio file
        convert_transformer = sox.Transformer()
        kwargs = dict(
            input_filepath=str(input_path),
            output_filepath=str(output_path),
            return_output=True,
        )
        if sample_rate:
            # we're only in this branch if the input file is raw
            convert_transformer.set_input_format(
                file_type="raw",
                rate=sample_rate,
                bits=16,
                channels=1,
                encoding="signed-integer",
            )
        status = convert_transformer.build(**kwargs)
        if status[0]:
            self.logger.warn(f"stdout: {status[1]}")
            self.logger.warn(f"stderr: {status[2]}")

    def _validate_extension(self, file_path: Path, file_type: str) -> bool:
        """
        checks if given file path has the specified extension

        Parameters
        ----------
        file_path : Path
            file path where the extension should be checked
        file_type : str
            the extension to validate with the given file, including the `.`

        Returns
        -------
        bool
            whether or not the file's extension matches the given extension.
        """
        if not file_path.suffix == file_type:
            return False
        else:
            return True

    def _trim_raw(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        sample_rate: int = 16000,
    ):
        """
        trims samples from the beginning and end of the specified audio file.
        amount of trimming is specified by precise values in seconds. see:
        https://pysox.readthedocs.io/en/latest/api.html?#sox.transform.Transformer.trim
        for more information

        Parameters
        ----------
        input_path : Path
            file path to the file that should be trimmed
        output_path : Path
            file path where the trimmed audio should be written
        start_time : float
            time in seconds after which audio should be _kept_
        end_time : float
            time in seconds after which audio should be _removed_
        sample_rate : int, optional
            the sample rate of the raw (pcm) audio file that will be trimmed,
            by default 16000
        """
        trim_transformer = sox.Transformer()
        trim_transformer.trim(start_time, end_time)
        trim_transformer.set_input_format(
            file_type="raw",
            rate=sample_rate,
            bits=16,
            channels=1,
            encoding="signed-integer",
        )
        kwargs = dict(
            input_filepath=str(input_path),
            output_filepath=str(output_path),
            return_output=True,
        )
        status = trim_transformer.build(**kwargs)
        if status[0]:
            self.logger.warn(f"stdout: {status[1]}")
            self.logger.warn(f"stderr: {status[2]}")

    def _pad_raw(
        self,
        input_path: Path,
        output_path: Path,
        start_pad: float = 0.0,
        end_pad: float = 0.0,
        sample_rate: int = 16000,
    ):
        """
        pads an audio file at the beginning and end. amount of padding is specified
        in precise values in seconds. see:
        https://pysox.readthedocs.io/en/latest/api.html?#sox.transform.Transformer.pad

        Parameters
        ----------
        input_path : Path
            file path to the file that should be padded
        output_path : Path
            file path where the padded audio should be written
        start_pad : float, optional
            length of silence in seconds to be added to the beginning of the file,
            by default 0.0
        end_pad : float, optional
            length of silence in seconds to be added to the end of the file, by
            default 0.0
        sample_rate : int, optional
            the sample rate of the raw (pcm) audio file that will be trimmed,
             by default 16000
        """
        # start_pad and end_pad in seconds only, unfortunately
        pad_transformer = sox.Transformer()
        pad_transformer.pad(start_duration=start_pad, end_duration=end_pad)
        pad_transformer.set_input_format(
            file_type="raw",
            rate=sample_rate,
            bits=16,
            channels=1,
            encoding="signed-integer",
        )
        kwargs = dict(
            input_filepath=str(input_path),
            output_filepath=str(output_path),
            return_output=True,
        )
        status = pad_transformer.build(**kwargs)
        if status[0]:
            self.logger.warn(f"stdout: {status[1]}")
            self.logger.warn(f"stderr: {status[2]}")

    def wav_to_pcm(self, wav_path: Path, pcm_path: Path):
        """
        converts the file given at `wav_path` to a raw (pcm) file and writes the result
        to `pcm_path`

        Parameters
        ----------
        wav_path : Path
            file path to the .wav file that should be converted to raw (pcm)
        pcm_path : Path
            file path where the converted audio should be written

        Raises
        ------
        ValueError
            if input or output file extensions are incorrect
        """
        valid_wav = self._validate_extension(wav_path, self.wav)
        valid_pcm = self._validate_extension(pcm_path, self.raw)
        if valid_wav and valid_pcm:
            self._convert(wav_path, pcm_path)
        else:
            raise ValueError(
                f"valid wav name: {valid_wav}:\n  {wav_path}\n"
                f"valid pcm name: {valid_pcm}:\n  {pcm_path}\n"
            )

    def pcm_to_wav(self, pcm_path: Path, wav_path: Path, sample_rate: int):
        """
        converts the file given at `pcm_path` to a .wav file and writes teh result to
        `wav_path`

        Parameters
        ----------
        pcm_path : Path
            file path to the raw (pcm) file that should be converted to .wav
        wav_path : Path
            file path where the converted audio should be written
        sample_rate : int
            the sample rate of the raw (pcm) input file

        Raises
        ------
        ValueError
            if the input or output file extensions are incorrect
        """
        valid_pcm = self._validate_extension(pcm_path, self.raw)
        valid_wav = self._validate_extension(wav_path, self.wav)
        if valid_pcm and valid_wav:
            self._convert(pcm_path, wav_path, sample_rate)
        else:
            raise ValueError(
                f"valid wav name: {valid_wav}:\n  {wav_path}\n"
                f"valid pcm name: {valid_pcm}:\n  {pcm_path}\n"
            )

    def select_channel(
        self, input_path: Path, output_path: Path, channel: int
    ) -> Tuple[str, str]:
        """
        creates a new wav file containing only the specified channel from the
        input file.

        Parameters
        ----------
        input_path : Path
            file path to the file containing the desired audio info
        output_path : Path
            file path where the extracted channel should be written
        channel : int
            1-based channel number specifying the channel to be extracted from
            the input file

        Returns
        -------
        Tuple[str, str]
            `stdout` and `stderr` as returned by sox
        """
        # creates a new wav file containing only the specified channel from the
        # input file.
        select_transformer = sox.Transformer()
        remix = {1: [channel]}
        # this api is a little different---calling `.remix()` returns a new
        # transformer
        selector = select_transformer.remix(remix)
        status = selector.build_file(
            input_filepath=str(input_path),
            output_filepath=str(output_path),
            return_output=True,
        )
        if status[0]:
            self.logger.warn(f"stdout: {status[1]}")
            self.logger.warn(f"stderr: {status[2]}")
        return status

    def trim_pcm(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        sample_rate: int = 16000,
    ):
        """
        validates raw (pcm) file extensions for both input and output paths before
        trimming samples from the beginning and end of the specified audio file. amount
        of trimming is specified in seconds.

        Parameters
        ----------
        input_path : Path
            file path to the file that should be trimmed
        output_path : Path
            file path where the trimmed audio should be written
        start_time : float
            time in seconds after which audio should be _kept_
        end_time : float
            time in seconds after which audio should be _removed_
        sample_rate : int, optional
            the sample rate of the raw (pcm) audio file that should be trimmed,
            by default 16000

        Raises
        ------
        ValueError
            if either the input or output files have an incorrect extension
        """
        # to trim only from the start of the file, specify None for `end time`
        valid_pcm_input = self._validate_extension(input_path, self.raw)
        valid_pcm_output = self._validate_extension(output_path, self.raw)
        if valid_pcm_input and valid_pcm_output:
            self._trim_raw(
                input_path, output_path, start_time, end_time, sample_rate=sample_rate
            )
        else:
            raise ValueError(
                f"valid input name: {valid_pcm_input}:\n {input_path}\n"
                f"valid output name: {valid_pcm_output}:\n {output_path}"
            )

    def pad_pcm(
        self,
        input_path: Path,
        output_path: Path,
        start_pad: float = 0.0,
        end_pad: float = 0.0,
        sample_rate: int = 16000,
    ):
        """
        validates raw (pcm) file extensions for input and outputh paths before padding
        an audio file at the beginning and end. amount of padding is specified in
        precise values in seconds.

        Parameters
        ----------
        input_path : Path
            file path to the file that should be padded
        output_path : Path
            file path where the padded audio should be written
        start_pad : float, optional
            length of the silence in seconds to be added to the beginning of the file,
            by default 0.0
        end_pad : float, optional
            length of the silence in seconds to be added to the end of the file,
            by default 0.0
        sample_rate : int, optional
            sample rate of the raw (pcm) audio file that will be trimmed,
            by default 16000

        Raises
        ------
        ValueError
            if either the input or the output files have an incorrect extension
        """
        valid_pcm_input = self._validate_extension(input_path, self.raw)
        valid_pcm_output = self._validate_extension(output_path, self.raw)
        if valid_pcm_input and valid_pcm_output:
            self._pad_raw(
                input_path, output_path, start_pad, end_pad, sample_rate=sample_rate
            )
        else:
            raise ValueError(
                f"valid input name: {valid_pcm_input}:\n {input_path}\n"
                f"valid output name: {valid_pcm_output}:\n {output_path}"
            )


class Processor:
    """
    a class that has some useful utilities for calling CLI tools and checking the
    results of those calls. parent to `Resampler` and `LevelMeter`
    """

    @staticmethod
    def _call_command(command: list, shell=False) -> tuple:
        """Calls external command and returns stdout and stderr results.

        Parameters
        ----------
        command : iterable
            An iterable containing the individual, space-delimited
            subcommands within a given cli command. In other words,
            a call to `cli_util --arg1 val1 --arg2 val2` would
            be represented as::
            ['cli_util', '--arg1', 'val1', '--arg2', 'val2']

        shell : bool, optional
            Must be true if command is a built-in shell command on Windows,
            by default False

        Returns
        -------
        tuple
            a tuple containing `stderr` and `stdout` results of the call
        """
        if sys.platform != "win32":
            shell = False  # ensure shell is false everywhere but windows
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
        )
        return process.communicate()

    @staticmethod
    def _calculate_filesize_ratio(in_path: Path, out_path: Path) -> float:
        """
        calculates the ratio of the size of the input file to the size of the output
        file. useful for determining if a resampling or copy operation are successful.

        Parameters
        ----------
        in_path : Path
            file path to the file whose size will be used in the numerator of the ratio
        out_path : Path
            file path to the file whose size will be used in the denominator of the
            ratio

        Returns
        -------
        float
            the ratio of the size of the input file to the size of the output file
        """
        # if either of these somehow don't exist at this point, we should
        # get a nice exception when we try to stat the file
        in_size = in_path.stat().st_size
        out_size = out_path.stat().st_size
        return in_size / out_size

    @staticmethod
    def _get_sample_length(
        sample_path: Path, bits_per_sample: int, sample_rate: int
    ) -> float:
        """
        calculates the length of a raw (pcm) file in seconds based on file size, number
        of bits per sample, and sample rate

        Parameters
        ----------
        sample_path : Path
            file path to file whose audio length should be measured
        bits_per_sample : int
            number of bits per sample, typically 16
        sample_rate : int
            sample rate of the specified raw (pcm) file

        Returns
        -------
        float
            length of the specified raw (pcm) audio file in seconds
        """
        # stat returns the size in bytes, so we have to divide
        # the number of bits by 8
        bytes_per_sample = bits_per_sample / 8
        return sample_path.stat().st_size / bytes_per_sample / sample_rate

    def _check_filesize_ratio(
        self, in_path: Path, out_path: Path, target_ratio: float
    ) -> bool:
        """
        checks if the ratio of the size of the input file to the size of the output
        file is close to the target ratio

        Parameters
        ----------
        in_path : Path
            file path to the file whose size will be used in the numerator of the ratio
        out_path : Path
            file path to the file whose size will be used in the denominator of the
            ratio
        target_ratio : float
            the expected ratio of the input file's size to the output file's size

        Returns
        -------
        bool
            whether or not the target ratio is close to the actual ratio

        Raises
        ------
        RuntimeError
            if the ratio doesn't match, which is a...choice to be sure
        """
        size_ratio = self._calculate_filesize_ratio(in_path, out_path)
        if np.isclose(size_ratio, target_ratio):
            return True
        else:
            raise RuntimeError(f"size ratio is {size_ratio}, expected {target_ratio}")


class Resampler(Processor):
    """
    a Processor that wraps the STL `filter` executable to resample raw (pcm) audio
    """

    # have to calculate seconds-to-trim in target sample rate because that's where
    # i measured STL filter delay
    pad_trim_seconds = {
        48000: 28 / 16000,
        32000: 31 / 16000,
        24000: 48 / 16000,
        16000: 0 / 16000,
        8000: 58 / 16000,
    }

    def __init__(self, path_to_filter: Path) -> None:
        """
        initializer for the Resampler class. stores the path to `filter` and creates
        a map from input sample rate to the appropriate downsampling function. also
        creates a `SoxConverter` instance.

        Parameters
        ----------
        path_to_filter : Path
            full file path to the `filter` executable
        """
        super().__init__()
        self.filter_path = str(path_to_filter)
        self.resampler_map = {
            48000: self.down_48k_to_16k,
            32000: self.down_32k_to_16k,
            24000: self.down_24k_to_16k,
            16000: self._copy_file,  # a little wasteful
            8000: self.up_8k_to_16k,
        }
        self.sox_converter = SoxConverter()

    def _copy_file(self, input_path: Path, output_path: Path) -> bool:
        """
        copies the file at `input_path` to `output_path`.

        Parameters
        ----------
        input_path : Path
            file path to be copied to `output_path`
        output_path : Path
            file path where `input_path` should be copied

        Returns
        -------
        bool
            returns true unless `shutil.copy` raises an exception...which is a choice
            to be sure
        """
        shutil.copy(input_path, output_path)
        return True

    def _call_filter(
        self, command: list, target_ratio: float, in_path: Path, out_path: Path
    ) -> bool:
        """
        calls the STL `filter` binary and checks if resampling was successful by
        comparing the size ratio of the input and output paths to the target
        ratio.

        Parameters
        ----------
        command : list
            the filter command itself and a list of arguments appropriate for the
            resampling to be performed
        target_ratio : float
            the expected ratio of the input file's size to the output file's size
        in_path : Path
            file path to the file that will be resampled
        out_path : Path
            file path where the resampled file should be written

        Returns
        -------
        bool
            whether or not the calculated file size ratio matches the target file size
            ratio, and by proxy indicating if resampling was successful
        """
        # `filter` puts all of its output in `stderr`, hmm
        stdout, stderr = self._call_command(command)

        # because we can't naively check `stderr`, let's naively calculate size ratio
        return self._check_filesize_ratio(in_path, out_path, target_ratio)

    def down_48k_to_16k(self, in_path: Path, out_path: Path) -> bool:
        """
        resamples the input raw (pcm) file, with sample rate assumed to be 48,000
        samp / sec to 16,000 samp / sec and writes the result.

        Parameters
        ----------
        in_path : Path
            file path to the raw (pcm) file with a sample rate of 48,000 samp / sec
        out_path : Path
            file path where the resampled raw (pcm) file should be written

        Returns
        -------
        bool
            whether the resampling operation was successful
        """
        target_ratio = 3.0
        command = [
            self.filter_path,
            "-q",
            "-down",
            "HQ3",
            str(in_path),
            str(out_path),
        ]

        return self._call_filter(command, target_ratio, in_path, out_path)

    def down_32k_to_16k(self, in_path: Path, out_path: Path) -> bool:
        """
        resamples the input raw (pcm) file, with sample rate assumed to be 32,000
        samp / sec to 16,000 samp / sec and writes the result.

        Parameters
        ----------
        in_path : Path
            file path to the raw (pcm) file with a sample rate of 32,000 samp / sec
        out_path : Path
            file path where the resampled raw (pcm) file should be written

        Returns
        -------
        bool
            whether the resampling operation was successful
        """
        target_ratio = 2.0
        command = [
            self.filter_path,
            "-q",
            "-down",
            "HQ2",
            str(in_path),
            str(out_path),
        ]

        return self._call_filter(command, target_ratio, in_path, out_path)

    def down_24k_to_16k(self, in_path: Path, out_path: Path) -> bool:
        """
        resamples the input raw (pcm) file, with sample rate assumed to be 24,000
        samp / sec to 16,000 samp / sec and writes the result.

        Parameters
        ----------
        in_path : Path
            file path to the raw (pcm) file with a sample rate of 24,000 samp / sec
        out_path : Path
            file path where the resampled raw (pcm) file should be written

        Returns
        -------
        bool
            whether the resampling operation was successful
        """
        with tempfile.NamedTemporaryFile() as temporary:
            temp_path = Path(temporary.name)
            target_ratio = 0.5
            upsample_command = [
                self.filter_path,
                "-q",
                "-up",
                "HQ2",
                str(in_path),
                str(temp_path),
            ]
            achieved_ratio = self._call_filter(
                upsample_command, target_ratio, in_path, temp_path
            )
            if not achieved_ratio:
                raise RuntimeError("intermediary upsampling failed")
            return self.down_48k_to_16k(temp_path, out_path)

    def up_8k_to_16k(self, in_path: Path, out_path: Path) -> bool:
        """
        resamples the input raw (pcm) file, with sample rate assumed to be 8,000
        samp / sec to 16,000 samp / sec and writes the result.

        Parameters
        ----------
        in_path : Path
            file path to the raw (pcm) file with a sample rate of 8,000 samp / sec
        out_path : Path
            file path where the resampled raw (pcm) file should be written

        Returns
        -------
        bool
            whether the resampling operation was successful
        """
        target_ratio = 0.5
        command = [
            self.filter_path,
            "-q",
            "-up",
            "HQ2",
            str(in_path),
            str(out_path),
        ]

        return self._call_filter(command, target_ratio, in_path, out_path)

    def resample_raw(
        self, input_path: Path, output_path: Path, input_sample_rate: int
    ) -> bool:
        """
        resamples an input file to 16 kHz and accounts for non-zero delay in the STL
        `filter` binary. returns true if successful.

        Parameters
        ----------
        input_path : Path
            file path to the raw (pcm) file that will be resampled
        output_path : Path
            file path where the resampled raw (pcm) file should be written
        input_sample_rate : int
            the sample rate of the file located at `input_path`

        Returns
        -------
        bool
            whether or not resampling was successful

        Raises
        ------
        RuntimeError
            if the file size ratio between the input file and the output file does
            not match the target ratio, ergo indicating whether resampling was
            successful
        """

        # zero pad the end of the input file, resample, then trim zeros from the front.
        with tempfile.TemporaryDirectory() as temporary:
            temp_path = Path(temporary)
            padded_path = temp_path / "padded.raw"
            resampled_path = temp_path / "resampled.raw"
            # pad
            self.sox_converter.pad_pcm(
                input_path,
                padded_path,
                0,
                self.pad_trim_seconds[input_sample_rate],
                input_sample_rate,
            )
            # resample
            resample_success = self.resampler_map[input_sample_rate](
                padded_path, resampled_path
            )
            if not resample_success:
                raise RuntimeError("intermediary resampling failed")
            # trim
            # don't need to pass sample rate in here because we're at 16k by now
            self.sox_converter.trim_pcm(
                resampled_path,
                output_path,
                self.pad_trim_seconds[input_sample_rate],
                None,
            )
        return resample_success


class LevelMeter(Processor):
    """
    a Processor that wraps the STL `actlev` executable to measure the active speech
    level of raw (pcm) audio
    """

    """
    example CLI output:
    | Processing
    -------------------------------------------------------
    Input file: ................... blah.raw, 16 bits, fs=16000 Hz
    Block Length: .................     256 [samples]
    Starting Block: ...............       1 []
    Number of Blocks: .............     608 []
    dBov desired for output: ...... -26.000 [dBov]
    Norm factor desired is: .......   1.219 [times]
    Max norm WITHOUT saturation: .. -11.691 [dBov]
    -------------------------------------------------------
    DC level: .....................      -2 [PCM]
    Maximum positive value: .......    5176 [PCM]
    Maximum negative value: .......   -4342 [PCM]
    -------------------------------------------------------
    Long term energy (rms): ....... -28.279 [dBov]
    Active speech level: .......... -27.721 [dBov]
    RMS peak-factor found: ........  12.249 [dB]
    Active peak factor found: .....  11.691 [dB]
    Activity factor: ..............  87.942 [%]
    -------------------------------------------------------%
    """

    # keys that should be converted to `int` after being read from CLI output
    int_keys = {"Samples:", "Min:", "Max:"}

    # names given to output values that we are interested in
    activity_factor = "%Active:"
    active_speech_level = "ActLev[dB]:"

    def __init__(self, path_to_actlev: Path) -> None:
        """
        initializer for the LevelMeter class. stores the path to `actlev` and the
        default block length used in the algorithm

        Parameters
        ----------
        path_to_actlev : Path
            full file path to the `actlev` executable
        """
        super().__init__()
        self.actlev_path = str(path_to_actlev)
        self.block_length = 256

    def _check_end_seconds(
        self, end_sec: float, sample_path: Path, bits: int, sample_rate: int
    ) -> float:
        """
        returns the minimum of the specified `end_sec` and the actual length in seconds
        of the file found at `sample_path`

        Parameters
        ----------
        end_sec : float
            a time in seconds
        sample_path : Path
            file path to the raw (pcm) file whose length in seconds should be tested
            against `end_sec`
        bits : int
            the number of bits per sample in the raw (pcm) file, typically 16
        sample_rate : int
            the sample rate associated with the raw (pcm) file

        Returns
        -------
        float
            the minimum of `end_sec` and the actual length in seconds of the file found
            at `sample_path`
        """
        end_sec = min(end_sec, self._get_sample_length(sample_path, bits, sample_rate))
        return end_sec

    def _seconds_to_blocks(
        self, start_sec: float, end_sec: float, sample_rate: int
    ) -> Tuple[int, int]:
        """
        converts a start and end time (in seconds) to a corresponding start and end
        block, which is how `actlev` expects start and end points to be communicated

        Parameters
        ----------
        start_sec : float
            desired start time in seconds
        end_sec : float
            desired end time in seconds
        sample_rate : int
            sample rate of the raw (pcm) file being processed

        Returns
        -------
        Tuple[int, int]
            the start and end block, respectively
        """

        # calculate conversion factor
        blocks_per_second = self.block_length / sample_rate

        # calculate block number—use floor to avoid picking a block number
        # that's higher than the total number of blocks in the file
        start_block = int(np.floor(start_sec / blocks_per_second))
        end_block = int(np.floor(end_sec / blocks_per_second))

        # `actlev` indexes blocks starting at 1, not 0
        if start_block == 0:
            start_block += 1

        return start_block, end_block

    def _parse_stdout(self, stdout: str) -> Dict[str, Any]:
        """
        parses the output generated by `actlev` into a dictionary

        Parameters
        ----------
        stdout : str
            string representation of the output generated by `actlev`

        Returns
        -------
        Dict[str, Any]
            names and values reported by `actlev`
        """

        # -q approach—separate by spaces first
        output = stdout.split()

        # calculate indices for keys and values
        key_inds = slice(0, len(output) - 1, 2)
        val_inds = slice(1, len(output), 2)

        # create dict and convert values appropriately
        results = dict(zip(output[key_inds], output[val_inds]))
        # convert string values to the appropriate numeric types
        results = {
            key: (int(val) if key in self.int_keys else float(val))
            for key, val in results.items()
        }

        return results

    def measure(
        self,
        speech_path: Path,
        sample_rate=16000,
        bits=16,
        target_level=-26,
        start_sec: float = None,
        end_sec: float = None,
    ) -> Tuple[float, float]:
        """
        measures the active speech level and the speach activity factor of the given
        raw (pcm) audio file

        Parameters
        ----------
        speech_path : Path
            file path to the raw (pcm) audio file to be measured
        sample_rate : int, optional
            sample rate associated with the raw (pcm) audio file,
            by default 16000
        bits : int, optional
            number of bits that represent each sample in the raw (pcm) audio file,
            by default 16
        target_level : int, optional
            target active speech level measured in dBov, by default -26
        start_sec : float, optional
            time in seconds at which to start measurements, by default None
        end_sec : float, optional
            time in seconds at which to end measurements, by default None

        Returns
        -------
        Tuple[float, float]
            the active speech level and the speech activity factor, respectively

        Raises
        ------
        RuntimeError
            if `actlev` reports something in `stderr`
        """

        command = [self.actlev_path]

        # specify start and end blocks if necessary
        if start_sec is not None and end_sec is not None:
            # NOTE: if you specify a block length beyond the length of the
            #       file in question, `actlev` will report:
            #       [filename]: Success
            #       in `stderr`
            #       so check for what the max block length should be, and
            #       report an issue on the STL github.
            end_sec = self._check_end_seconds(end_sec, speech_path, bits, sample_rate)
            start_block, end_block = self._seconds_to_blocks(
                start_sec, end_sec, sample_rate
            )
            command.extend(["-start", str(start_block), "-end", str(end_block)])

        # build up CLI args
        command.extend(
            [
                "-sf",
                str(sample_rate),
                "-bits",
                str(bits),
                "-lev",
                str(target_level),
                "-q",
                str(speech_path),
            ]
        )
        # actually call the CLI
        stdout, stderr = self._call_command(command)
        if stderr:
            raise RuntimeError(f"uuhhh: {stderr}")
        # parse the result and return the items we're interested in
        stdout = stdout.decode()
        results = self._parse_stdout(stdout)
        active_speech_level = results[self.active_speech_level]
        activity_factor = results[self.activity_factor]
        return active_speech_level, activity_factor


class SpeechNormalizer(Processor):
    """
    a Processor that wraps the STL `sv56demo` executable to normalise the active
    speech level in a raw (pcm) audio file to the specified level
    """

    # Usage:
    # $ sv56demo [-options] FileIn FileOut
    #            [BlockSize [1stBlock [NoOfBlocks [DesiredLevel
    #            [SampleRate [Resolution] ] ] ] ] ]

    def __init__(self, path_to_sv56demo: Path) -> None:
        """Initializer for the SpeechNormalizer class. stores the path to `sv56demo`"""
        super().__init__()
        self.path_to_sv56demo = str(path_to_sv56demo)

    def normalizer(
        self,
        in_path: Path,
        out_path: Path,
        sample_rate=16000,
        bits=16,
        target_level=-26,
    ) -> bool:
        """
        normalizes a given raw (pcm) audio file to a target active speech level

        Parameters
        ----------
        in_path : Path
            file path to the raw (pcm) audio file to be normalized
        out_path : Path
            file path where the normalized raw (pcm) audio file should be written
        sample_rate : int, optional
            sample rate associated with the raw (pcm audio file),
            by default 16000
        bits : int, optional
            number of bits that represent each samel in the raw (pcm) audio file,
            by default 16
        target_level : int, optional
            target active speech level measured in dBov, by default -26

        Returns
        -------
        bool
            whether or not a file has been written to `out_path` that has the same
            file size as the file at `in_path`, a crude proxy for checking that all
            steps completed successfully
        """

        # set up the CLI command
        target_ratio = 1.0
        sample_rate = str(sample_rate)
        bits = str(bits)
        target_level = str(target_level)
        command = [
            self.path_to_sv56demo,
            "-lev",
            target_level,
            "-sf",
            sample_rate,
            "-bits",
            bits,
            "-qq",
            str(in_path),
            str(out_path),
        ]

        # `filter` puts all of its output in `stderr`
        stdout, stderr = self._call_command(command)

        # because we can't naively check `stderr`, let's naively calculate size ratio
        return self._check_filesize_ratio(in_path, out_path, target_ratio)


@click.command()
@click.argument("infile", type=click.Path(exists=True))
@click.argument("outfile", type=click.Path())
@click.argument("inrate", type=click.INT)
def resample(infile: str, outfile: str, inrate: int):
    """
    CLI utility that performs the somewhat-tedious process of resampling a .wav file

    Parameters
    ----------
    infile : str
        file path to the .wav file to be resampled
    outfile : str
        file path where the resampled .wav file should be written
    inrate : int
        sample rate of the input .wav file, required because i haven't implemented
        reading the sample rate from the .wav file yet
    """
    infile = Path(infile)
    outfile = Path(outfile)

    # read some config
    stl_path = Path(get_stl_path())

    # build our resampler and sox converter
    resampler = Resampler(stl_path / "filter")
    converter = SoxConverter()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # 1. convert to raw
        in_raw_path = temp_path / f"{infile.stem}.raw"
        converter.wav_to_pcm(infile, in_raw_path)
        # 2. resample
        resampled_raw_path = temp_path / f"{infile.stem}_16000.raw"
        success = resampler.resample_raw(in_raw_path, resampled_raw_path, inrate)
        if not success:
            RuntimeError(f"failed resampling {infile}")
        # 3. convert to wav
        resampled_wav_path = temp_path / f"{infile.stem}_16000.wav"
        converter.pcm_to_wav(resampled_raw_path, resampled_wav_path, 16000)
        # 4. copy to outfile
        resampler._copy_file(resampled_wav_path, outfile)


if __name__ == "__main__":
    resample()
