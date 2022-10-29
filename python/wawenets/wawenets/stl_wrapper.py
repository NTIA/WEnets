import tempfile
import subprocess
import sys

from pathlib import Path
from typing import Tuple

import sox

import numpy as np

from wawenets.generic_logger import construct_logger


class FileWalker:
    def __init__(self):
        pass

    def get_subdirectories(self, directory: Path):
        return [item for item in directory.iterdir() if item.is_dir()]

    def get_files_with(self, directory: Path, match: str):
        return directory.glob(f"*{match}")


class SoxConverter:
    wav = ".wav"
    raw = ".raw"

    def __init__(self):
        # loggggggg
        self.logger = construct_logger(self.__class__.__name__)

    def _convert(self, input_path: Path, output_path: Path, sample_rate: int = None):
        """assumes input path is a one-channel audio file"""
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

    def _validate_extension(self, file_path: Path, file_type: str):
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

    def wav_to_pcm(self, wav_path: Path, pcm_path: Path):
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
        valid_pcm = self._validate_extension(pcm_path, self.raw)
        valid_wav = self._validate_extension(wav_path, self.wav)
        if valid_pcm and valid_wav:
            self._convert(pcm_path, wav_path, sample_rate)
        else:
            raise ValueError(
                f"valid wav name: {valid_wav}:\n  {wav_path}\n"
                f"valid pcm name: {valid_pcm}:\n  {pcm_path}\n"
            )

    def select_channel(self, input_path: Path, output_path: Path, channel: int):
        """creates a new wav file containing only the specified channel from the
        input file."""
        select_transformer = sox.Transformer()
        remix = {1: [channel]}
        # this api is a little different---calling `.remix()` returns a new
        # transformer
        selector = select_transformer.remix(remix)
        status = selector.build_file(
            input_filepath=str(input_path), output_filepath=str(output_path)
        )
        return status

    def trim_pcm(
        self, input_path: Path, output_path: Path, start_time: float, end_time: float
    ):
        valid_pcm_input = self._validate_extension(input_path, self.raw)
        valid_pcm_output = self._validate_extension(output_path, self.raw)
        if valid_pcm_input and valid_pcm_output:
            self._trim_raw(input_path, output_path, start_time, end_time)
        else:
            raise ValueError(
                f"valid input name: {valid_pcm_input}:\n {input_path}\n"
                f"valid output name: {valid_pcm_output}:\n {output_path}"
            )


class Processor:
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
        # if either of these somehow don't exist at this point, we should
        # get a nice exception when we try to stat the file
        in_size = in_path.stat().st_size
        out_size = out_path.stat().st_size
        return in_size / out_size

    @staticmethod
    def _get_sample_length(sample_path: Path, bits: int, sample_rate: int):
        # stat returns the size in bytes, so we have to divide
        # the number of bits by 8
        bytes_per_sample = bits / 8
        return sample_path.stat().st_size / bytes_per_sample / sample_rate

    def _check_filesize_ratio(
        self, in_path: Path, out_path: Path, target_ratio: float
    ) -> bool:

        size_ratio = self._calculate_filesize_ratio(in_path, out_path)
        if np.isclose(size_ratio, target_ratio):
            return True
        else:
            raise RuntimeError(f"size ratio is {size_ratio}, expected {target_ratio}")


class Resampler(Processor):
    def __init__(self, path_to_filter: Path) -> None:
        super().__init__()
        self.filter_path = str(path_to_filter)

    def _call_filter(
        self, command: list, target_ratio: float, in_path: Path, out_path: Path
    ) -> bool:
        # `filter` puts all of its output in `stderr`, hmm
        stdout, stderr = self._call_command(command)

        # because we can't naively check `stderr`, let's naively calculate size ratio
        return self._check_filesize_ratio(in_path, out_path, target_ratio)

    def down_48k_to_16k(self, in_path: Path, out_path: Path) -> bool:
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


class LevelMeter(Processor):
    """| Processing
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

    int_keys = {"Samples:", "Min:", "Max:"}

    activity_factor = "%Active:"
    active_speech_level = "ActLev[dB]:"

    def __init__(self, path_to_actlev: Path) -> None:
        super().__init__()
        self.actlev_path = str(path_to_actlev)
        self.block_length = 256

    def _check_end_seconds(
        self, end_sec: float, sample_path: Path, bits: int, sample_rate: int
    ):
        # calculate reasonable end point
        end_sec = min(end_sec, self._get_sample_length(sample_path, bits, sample_rate))
        return end_sec

    def _seconds_to_blocks(
        self, start_sec: float, end_sec: float, sample_rate: int
    ) -> tuple:

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

    def _parse_stdout(self, stdout: str) -> Tuple:

        # -q approach—separate by spaces first
        output = stdout.split()

        # calculate indices for keys and values
        key_inds = slice(0, len(output) - 1, 2)
        val_inds = slice(1, len(output), 2)

        # create dict and convert values appropriately
        results = dict(zip(output[key_inds], output[val_inds]))
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
    ) -> Tuple:

        command = [self.actlev_path]

        # specify start and end blocks if necessary
        if start_sec is not None and end_sec is not None:
            # TODO: if you specify a block length beyond the length of the
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

        stdout, stderr = self._call_command(command)
        if stderr:
            raise RuntimeError(f"uuhhh: {stderr}")
        stdout = stdout.decode()
        results = self._parse_stdout(stdout)
        active_speech_level = results[self.active_speech_level]
        activity_factor = results[self.activity_factor]
        return active_speech_level, activity_factor


class SpeechNormalizer(Processor):
    """Usage:
    $ sv56demo [-options] FileIn FileOut
               [BlockSize [1stBlock [NoOfBlocks [DesiredLevel
               [SampleRate [Resolution] ] ] ] ] ]"""

    def __init__(self, path_to_sv56demo) -> None:
        super().__init__()
        self.path_to_sv56demo = path_to_sv56demo

    def normalizer(
        self,
        in_path: Path,
        out_path: Path,
        sample_rate=16000,
        bits=16,
        target_level=-26,
    ) -> bool:

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
