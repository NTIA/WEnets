# WAWEnets Matlab Code
Implements Wideband Audio Waveform Evaluation networks or WAWEnets.

This WAWEnets implementation produces one or more speech quality or intelligibility values for each input speech signal without using reference speech signals.
WAWEnets are convolutional neural networks and they have been trained using full-reference objective speech quality and speech intelligibility values.

Details can be found in [the WAWEnets paper.](https://www.its.bldrdoc.gov/publications/details.aspx?pub=3242)<sup id="wawenets">[1](#f1)</sup>

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

# How to run
* Clone to local repository  
* Open using Matlab or Octave  
* Run WAWEnet.m.

Many example calls and results are provided in `demo.m`.

Input can be a vector of speech samples, the name of a `.wav` file, or the name of a text file containing a list of `.wav` files.

Results are shown in the Matlab command window, are returned in a structure, and can also be logged into an optional output text file.

Further details below (or type `help WAWEnet` in the Matlab command window).

# Usage

```
[allFileInfo, ctlInfo] = WAWEnet(inSpeech, ctlInfo)
```

## Inputs

The only required input is `inSpeech`.  This may be a row or column vector of speech samples or a string of the form `*.wav` or `*.txt`.

If `inSpeech` is a vector of speech samples the sample rate must be 16,000 smp/sec, it must contain at least 48,000 samples (3 sec.) and the nominal range for the sample values is +/- 1.0.

To best match the designed scope of WAWEnets, it should have a speech activity factor of roughly 0.5 or greater and an active speech level near 26 dB below the clipping points of +/- 1.0.  (See level normalization feature below.)

If `inSpeech` is string that names a `.wav` file, that `.wav` file must:

- be uncompressed
- have sample rate 8, 16, 24, 32, or 48k smp/sec.
- contain at least 3 seconds of speech

To best match the designed scope of WAWEnets, the `.wav` file should have a speech activity factor of roughly 0.5 or greater and an active speech level near 26 dB below the clipping points of +/- 1.0.  (See level normalization feature below.) The native sample rate for WAWEnets is 16 k smp/sec so files with rates 8, 24, 32, or 48k rate are converted internally before processing.

If `inSpeech` is string that names a `.txt` file, each line should specify a `.wav` file that meets the `.wav` file requirements listed above. Each file will be processed in sequence. This will be slightly more efficient (1 or 2% reduction in run-time) than calling WAWEnet.m repeatedly. 

`ctlInfo` is an optional input. It is a structure that provides control information to `WAWEnet.m`. Any or all of the five different fields may be used in any combination as follows:

`ctlInfo.WAWEnetMode = M` invokes the WAWEnet trained using a specific full-reference target.

- `M = 1`: WAWEnet trained using WB-PESQ<sup id="wbpesq">[2](#f2)</sup> target values (Default)
- `M = 2`: WAWEnet trained using POLQA<sup id="polqa">[3](#f3)</sup> target values
- `M = 3`: WAWEnet trained using PEMO<sup id="pemo">[4](#f4)</sup> target values
- `M = 4`: WAWEnet trained using STOI<sup id="stoi">[5](#f5)</sup> target values

`ctlInfo.levelNormalization = L` specifies internal level normalization of the speech input (vector or `.wav` file) to 26 dB below clipping.
- `L = 0`: normalization off
- `L = 1`: normalization on (Default)

`ctlInfo.segmentStep = S` specifies the segment step (stride) and is an integer with value 1 or greater.  Default `S = 48,000`. WAWEnet requires a full 3 seconds of signal to generate a result.  If the input speech (vector or `.wav` file) is longer than 3 seconds multiple results may be produced. `S` specifies the number of samples to move ahead in the vector or file when extracting the next segment. The default value of 48,000 gives zero overlap between segments. Using this default any input less than 6 sec. will produce one result, based on just the first 3 sec. A 6 sec. input will produce two results. If `S = 24,000` for example, segment overlap will be 50%, a 4.5 sec. input will produce 2 results and a 6 sec. input will produce 3 results.

`ctlInfo.channel = C` specifies a channel number to use when the input speech is in a multi-channel `.wav` file. Default is `C = 1`.

`ctlInfo.outFilename = 'myFile.txt'` specifies a text file that captures WAWEnet results on a new line for each speech input processed. If the file exists it will be appended. The extension `.txt` will be added as needed. Default is that no `.txt` file is generated.

## Outputs

The output for each of the N speech signals processed is in the 1 by N structure `allFileInfo`. The fields are:

- `allFileInfo.name`, `.wav` filename or "MatlabVector"
- `allFileInfo.exception`, description of any exception encountered
- `allFileInfo.sampleRate`, native sample rate of the signal
- `allFileInfo.duration`, duration of the signal in seconds
- `allFileInfo.activeLevel`, active speech level in dB below overload
- `allFileInfo.activityFactor`, speech activity factor
- `allFileInfo.netOut`, output value produced by WAWEnet

`ctlInfo` is returned as well.  In addition to the five fields defined in the input section above, it also includes:
- `ctlInfo.sampleRate`, native sample rate for WAWEnets, always 16,000 smp/s
- `ctlInfo.segmentLength`, native segment length rate for WAWEnet, always 48,000 smp (3 s)
- `ctlInfo.outFileHeader`, the header that defines results shown on screen and in any output file.  This includes a starting time/date stamp.

-------------------------------------
<b id="f1">1</b> Andrew A. Catellier & Stephen D. Voran, "WAWEnets: A No-Reference Convolutional Waveform-Based Approach to Estimating Narrowband and Wideband Speech Quality," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 331-335. [↩](#wawenets)

<b id="f2">2</b> ITU-T Recommendation P.862, "Perceptual evaluation of speech quality (PESQ)," Geneva, 2001.[↩](#wbqesq)

<b id="f3">3</b> ITU-T Recommendation P.863, "Perceptual objective listening quality analysis," Geneva, 2018.[↩](#polqa)

<b id="f4">4</b> R. Huber and B. Kollmeier, "PEMO-Q — A new method for objective audio quality assessment using a model of auditory perception," IEEE Trans. ASLP, vol. 14, no. 6, pp. 1902-1911, Nov. 2006.[↩](#pemo)

<b id="f5">5</b> C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," IEEE Trans. ASLP, vol. 19, no. 7, pp. 2125-2136, Sep. 2011.[↩](#stoi)