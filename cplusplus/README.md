# WAWEnets C++ code

Implements Wideband Audio Waveform Evaluation networks or WAWEnets.

This WAWEnets implementation produces one or more speech quality or intelligibility values for each input speech signal without using reference speech signals.
WAWEnets are convolutional neural networks and they have been trained using full-reference objective speech quality and speech intelligibility values.

Details can be found in <a href="https://www.its.bldrdoc.gov/publications/3242.aspx" target="_blank">the WAWEnets paper.</a><sup id="wawenets">[1](#f1)</sup>

If you need to cite our work, please use the following:

```
@INPROCEEDINGS{
   9054204,
   author={A. A. {Catellier} and S. D. {Voran}},
   booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   title={Wawenets: A No-Reference Convolutional Waveform-Based Approach to Estimating Narrowband and Wideband Speech Quality},
   year={2020},
   volume={},
   number={},
   pages={331-335},
}
```

# Setup

This implementation depends on the Pytorch® C++ library and was built using Visual Studio® on Windows. The following steps are required to build the project:

1. Download the [Pytorch C++ Library](https://pytorch.org/get-started/locally/) by selecting `stable`, `Windows`, `LibTorch`, `C++/Java` and `10.2` and then clicking the provided download link.
2. Extract the downloaded `.zip` file to a directory on your computer.
3. Add `$PATH_TO_LIBTORCH\libtorch-win-shared-with-deps-1.5.0\libtorch\` to your `PATH` enviroment variables where `$PATH_TO_LIBTORCH` is the directory where you unzipped the library.
4. Open the `wawenet` project in Visual Studio.
5. In the Visual Studio Project properties, set the following options:
    - VC++ Directories->Include Directories-> `$PATH_TO_LIBTORCH\libtorch-win-shared-with-deps-1.5.0\libtorch\include`
    - VC++ Directories->Library Directories-> `$PATH_TO_LIBTORCH\libtorch-win-shared-with-deps-1.5.0\libtorch\lib`
    - C/C++ -> General -> Additional Include Directories -> `$PATH_TO_LIBTORCH\libtorch-win-shared-with-deps-1.5.0\libtorch\include`
    - Linker->General-> Additional Library Directories -> `$PATH_TO_LIBTORCH\libtorch-win-shared-with-deps-1.5.0\libtorch\lib`
    - Linker->Input>Additional Dependencies->
        - torch.lib
        - c10.lib
        - caffe2_nvrtc.lib
        - torch_cpu.lib
6. Copy these `.dll` files from `$PATH_TO_LIBTORCH\libtorch-win-shared-with-deps-1.5.0\libtorch\lib` to the `Release` directory (where the .exe lives)
    - torch.dll
    - c10.dll
    - caffe2_nvrtc.dll
    - torch_cpu.dll
8. The [`AudioFile`](https://github.com/adamstark/AudioFile) library is required to build the project. Place `AudioFile.h` in the `cplusplus/Wawenet` directory.
8. The WAWEnet PyTorch models are required to be in the same directory as the executable or in the working directory.
10. Set Visual Studio build settings to `x64 release` (building a debug executable requires downloading the debug version of PyTorch).
11. To run WAWEnet, see below or set command-line arguments in Visual Studio Properties -> Debugging -> Command Arguments.

# Usage

```
wawenet.exe infile [-m M|-l L|-s S|-c C|-o 'myFile.txt']
```

## Arguments

`infile` is either a .wav file or a .txt file where each line specifies a suitable .wav file. In this second case, the listed .wav files will be processed in sequence.

A suitable `.wav` file must:

- be uncompressed
- have sample rate 8, 16, 24, 32, or 48k smp/sec.
- contain at least 3 seconds of speech


To best match the designed scope of WAWEnets, the `.wav` file should have a speech activity factor of roughly 0.5 or greater and an active speech level near 26 dB below the clipping points of +/- 1.0 (see level normalization feature below). The native sample rate for WAWEnets is 16 k smp/sec so files with rates 8, 24, 32, or 48k rate are converted internally before processing.

`-m M` specifies a WAWEnet mode. The integer M specifies the WAWEnet trained using a specific full-reference target.

- `-m 1`: WAWEnet trained using WB-PESQ<sup id="wbpesq">[2](#f2)</sup> target values (Default)
- `-m 2`: WAWEnet trained using POLQA<sup id="polqa">[3](#f3)</sup> target values
- `-m 3`: WAWEnet trained using PEMO<sup id="pemo">[4](#f4)</sup> target values
- `-m 4`: WAWEnet trained using STOI<sup id="stoi">[5](#f5)</sup> target values

`-l L` specifies internal level normalization of `.wav` file contents to 26 dB below clipping.

- `-l 0`: normalization off
- `-l 1`: normalization on (Default)

`-s S` specifies specifies the segment step (stride) and is an integer with value 1 or greater.  Default is `-s 48,000`. WAWEnet requires a full 3 seconds of signal to generate a result.  If a `.wav` file is longer than 3 seconds multiple results may be produced. `S` specifies the number of samples to move ahead in the vector or file when extracting the next segment. The default value of 48,000 gives zero overlap between segments. Using this default any input less than 6 sec. will produce one result, based on just the first 3 sec. A 6 sec. input will produce two results. If `-s 24,000` for example, segment overlap will be 50%, a 4.5 sec. input will produce 2 results and a 6 sec. input will produce 3 results.

`-c C` specifies a channel number to use when the input speech is in a multi-channel `.wav` file. Default is `-c 1`.

`-o 'myFile.txt'` specifies a text file that captures WAWEnet results on a new line for each speech input processed. If the file exists it will be appended to. The extension `.txt` will be added as needed. Default is that no `.txt` file is generated.

## Outputs

The output for each of the N speech signals processed is in the format:

```
[wavfile] [channel] [sample_rate] [duration] [active_level] [speech_activity] [level_normalization] [segment_step_size] [WAWEnet_mode] [model_prediction]
```

where:

- `wavfile` is the filename that has been processed
- `channel`, is the channel of `wavfile` that has been processed
- `sample_rate`, native sample rate of the `wavfile`
- `duration`, duration of `wavfile` in seconds
- `active_level`, active speech level of the last processed segment of `wavfile` in dB below overload
- `speech_activity` is the speech activity factor of the last processed segment of `wavfile`
- `level_normalization` reflects whether `wavfile` was normalized during processing
- `segment_step_size` reflects the segment step (stride) used to process `wavfile`
- `WAWEnet_mode` is the mode `wavfile` has been processed with
- `model_prediction`, output value (or values, if >1 segment was analyzed) produced by WAWEnet for `wavfile`

-------------------------------------
<b id="f1">1</b> Andrew A. Catellier & Stephen D. Voran, "WAWEnets: A No-Reference Convolutional Waveform-Based Approach to Estimating Narrowband and Wideband Speech Quality," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 331-335. [↩](#wawenets)

<b id="f2">2</b> ITU-T Recommendation P.862, "Perceptual evaluation of speech quality (PESQ)," Geneva, 2001.[↩](#wbqesq)

<b id="f3">3</b> ITU-T Recommendation P.863, "Perceptual objective listening quality analysis," Geneva, 2018.[↩](#polqa)

<b id="f4">4</b> R. Huber and B. Kollmeier, "PEMO-Q — A new method for objective audio quality assessment using a model of auditory perception," IEEE Trans. ASLP, vol. 14, no. 6, pp. 1902-1911, Nov. 2006.[↩](#pemo)

<b id="f5">5</b> C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," IEEE Trans. ASLP, vol. 19, no. 7, pp. 2125-2136, Sep. 2011.[↩](#stoi)
