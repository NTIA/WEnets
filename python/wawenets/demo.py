import subprocess
import sys


def call_command(command: list, shell=False) -> tuple:
    """Calls external command and returns stdout and stderr results.

    Parameters
    ----------
    command : iterable
        An iterable containing the individual, space-delimited
        subcommands within a given call_command command. In other words,
        a call to `call_command_util --arg1 val1 --arg2 val2` would
        be represented as::
        ['call_command_util', '--arg1', 'val1', '--arg2', 'val2']

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


print(
    "This file demos WAWEnet functionality and provides some verification that the "
    "code has been compiled correctly."
)
print(
    "Assumes `wawenet.py` is in this directory, and that sample speech is located "
    "in ../../speech"
)
print("Results should match closely, but may not match exactly. This is expected.")

print("*****************************************************************")
# One .wav file
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000439_Q231_D401.wav"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000439_Q231_D401.wav:  1 16000 3.000000 -25.978893 0.495506 1 48000 1 [ 1.569181 ] 1.569181"
)

print("*****************************************************************")
# list of 10 .wav files, results to screen
std_out, _ = call_command(["python", "wawenets_cli.py", "-i", "list.txt"])
print(std_out.decode())

# list of 10 .wav files, results to screen and myText.txt
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "list.txt", "-o", "myText.txt"]
)
print(std_out.decode())

print("the output of both commands should be:")
print(
    "../../speech/T000053_Q159_D401.wav:  1 16000 3.000000 -26.034792 0.915251 1 48000 1 [ 1.222245 ] 1.222245"
)
print(
    "../../speech/T000093_Q446_D401.wav:  1 16000 3.000000 -25.992599 0.498567 1 48000 1 [ 4.194879 ] 4.194879"
)
print(
    "../../speech/T000342_Q125_D401.wav:  1 16000 3.000000 -25.999733 0.988531 1 48000 1 [ 1.181551 ] 1.181551"
)
print(
    "../../speech/T000439_Q231_D401.wav:  1 16000 3.000000 -25.978893 0.495506 1 48000 1 [ 1.569181 ] 1.569181"
)
print(
    "../../speech/T000493_Q415_D401.wav:  1 16000 3.000000 -25.993883 0.747076 1 48000 1 [ 2.939228 ] 2.939228"
)
print(
    "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 1 [ 3.675040 ] 3.675040"
)
print(
    "../../speech/T000937_Q340_D401.wav:  1 16000 3.000000 -25.992653 0.793695 1 48000 1 [ 2.467979 ] 2.467979"
)
print(
    "../../speech/T001002_Q366_D401.wav:  1 16000 3.000000 -25.985020 0.820408 1 48000 1 [ 3.668306 ] 3.668306"
)
print(
    "../../speech/T001121_Q269_D401.wav:  1 16000 3.000000 -25.985125 0.572003 1 48000 1 [ 2.789083 ] 2.789083"
)
print(
    "../../speech/T001189_Q200_D401.wav:  1 16000 3.000000 -25.983643 0.491644 1 48000 1 [ 2.211183 ] 2.211183"
)
print(
    "../../speech/long.wav:  1 16000 4.717500 -24.886742 0.807339 1 48000 1 [ 4.229949 ] 4.229949"
)


print("*****************************************************************")
# One .wav file, 4.7 sec long, results to screen, explicit selection of 4 inputs
# mode 2
# level normalization off
# segment step of 12000 gives 75% segment, overlap, will give 3 segments and 3 results
# channel 1 of .wav file explicitly selected
std_out, _ = call_command(
    [
        "python",
        "wawenets_cli.py",
        "-i",
        "../../speech/long.wav",
        "-m",
        "2",
        "-l",
        "False",
        "-s",
        "12000",
        "-c",
        "1",
    ]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/long.wav:  1 16000 4.717500 -27.232590 0.925942 0 12000 2 [ 4.466176 4.156981 4.396053  ]  4.339736"
)


print("*****************************************************************")
# One .wav file, ~10 sec long, with no speech, results to screen
std_out, _ = call_command(
    [
        "python",
        "wawenets_cli.py",
        "-i",
        "../../speech/long_silence.wav",
        "-m",
        "2",
        "-l",
        "False",
        "-s",
        "12000",
    ]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/long_silence.wav:  1 16000 10.025187 -100.000000 0.000000 0 12000 2 [ 2.832274 2.831977 2.830791 2.833440 2.833960 2.830233 2.828940 2.832784 2.830861 2.832243  ]  nan"
)


print("*****************************************************************")
# One .wav file with 2 channels.
# channel 1 of .wav file explicitly selected
# This channel is identical to ../speech/T000053_Q159_D401.wav
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/TwoChannel_53_93.wav", "-c", "1"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/TwoChannel_53_93.wav:  1 16000 3.000000 -26.034792 0.915251 1 48000 1 [ 1.222245 ] 1.222245"
)


print("*****************************************************************")
# channel 2 of .wav file explicitly selected
# This channel is identical to ../speech/T000093_Q446_D401.wav
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/TwoChannel_53_93.wav", "-c", "2"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/TwoChannel_53_93.wav:  2 16000 3.000000 -25.992599 0.498567 1 48000 1 [ 4.194879 ] 4.194879"
)


print("*****************************************************************")
# One .wav file, sample rate 8k results to screen only
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401_8.wav"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401_8.wav:  1 8000 3.000000 -25.597240 0.420938 1 48000 1 [ 2.835287 ] nan"
)


print("*****************************************************************")
# One .wav file, sample rate 16k results to screen only
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401.wav"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 1 [ 3.675040 ] 3.675040"
)


print("*****************************************************************")
# One .wav file, sample rate 24k results to screen only
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401_24.wav"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401_24.wav:  1 24000 3.000000 -26.271122 0.517105 1 48000 1 [ 3.604044 ] 3.604044"
)


print("*****************************************************************")
# One .wav file, sample rate 32k results to screen only
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401_32.wav"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401_32.wav:  1 32000 3.000000 -26.338856 0.530771 1 48000 1 [ 3.610513 ] 3.610513"
)


print("*****************************************************************")
# One .wav file, sample rate 48k results to screen only
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401_48.wav"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401_48.wav:  1 48000 3.000000 -26.394070 0.522247 1 48000 1 [ 3.740452 ] 3.740452"
)


print("*****************************************************************")
# One .wav file, mode 1
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401.wav", "-m", "1"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 1 [ 3.675040 ] 3.675040"
)


print("*****************************************************************")
# One .wav file, mode 2
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401.wav", "-m", "2"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 2 [ 3.584861 ] 3.584861"
)


print("*****************************************************************")
# One .wav file, mode 3
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401.wav", "-m", "3"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 3 [ 0.811744 ] 0.811744"
)


print("*****************************************************************")
# One .wav file, mode 4
std_out, _ = call_command(
    ["python", "wawenets_cli.py", "-i", "../../speech/T000863_Q316_D401.wav", "-m", "4"]
)
print(std_out.decode())
print("result should be: ")
print(
    "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 4 [ 0.988499 ] 0.988499"
)
