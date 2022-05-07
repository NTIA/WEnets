function [allFileInfo, ctlInfo] = WAWEnet(inSpeech,varargin)
%WAWEnet.m produces one or more speech quality or intelligibility values
%for each input speech signal. This is done without the use of any 
%reference speech signals. These no-reference values are created by
%Wideband Audio Waveform Evaluation networks or WAWEnets[1]. WAWEnets are
%convolutional neural networks and they have been trained using
%full-reference objective speech quality and speech intelligibility values.
%
%Use:  [allFileInfo, ctlInfo] = WAWEnet(inSpeech, ctlInfo)
%
%--------------------------------INPUTS--------------------------------
%The only required input is inSpeech.  This may be a row or column vector
%of speech samples or a string of the form *.wav or *.txt.
%
%If inSpeech is a vector of speech samples the sample rate must be
%16,000 smp/sec, it must contain at least 48,000 samples (3 sec.) and the
%nominal range for the sample values is +/- 1.0.
%
%To best match the designed scope of WAWEnets, it should have a speech
%activity factor of roughly 0.5 or greater and an active speech level near
%26 dB below the clipping points of +/- 1.0.  (See level normalization
%feature below.)
%
%If inSpeech is a string that names a .wav file, that file must: 
%   -be uncompressed
%   -have sample rate 8, 16, 24, 32, or 48k smp/sec.
%   -contain at least 3 seconds of speech
%
%To best match the designed scope of WAWEnets, the .wav file should have a
%speech activity factor of roughly 0.5 or greater and an active speech
%level near 26 dB below the clipping points of +/- 1.0.  (See level
%normalization feature below.) The native sample rate for WAWEnets is
%16 k smp/sec so files with rates 8, 24, 32, or 48k rate are converted
%internally before processing.
%
%If inSpeech is a string that names a .txt file, each line should specify
%a .wav file that meets the .wav file requirements listed above.
%Each file will be processed in sequence. This will be slightly more
%efficient (1 or 2% reduction in run-time) than calling WAWEnet.m
%repeatedly. 
%
%ctlInfo is an optional input.  It is a structure that provides control
%information to WAWEnet.m.  Any or all of the five different fields may be
%used in any combination as follows:
%
%ctlInfo.WAWEnetMode = M invokes the WAWEnet trained using a specific
%full-reference target.
%M = 1: WAWEnet trained using WB-PESQ [2] target values (Default)
%M = 2: WAWEnet trained using POLQA [3] target values
%M = 3: WAWEnet trained using PEMO [4] target values
%M = 4: WAWEnet trained using STOI [5] target values
%
%ctlInfo.levelNormalization = L specifies internal level normalization of
%the speech input (vector or .wav file) to 26 dB below clipping.
%L = 0: normalization off
%L = 1: normalization on (Default)
%
%ctlInfo.segmentStep = S specifies the segment step (stride) and is an
%integer with value 1 or greater.  Default is S = 48,000.
%WAWEnet requires a full 3 seconds of signal to generate a result. 
%If the input speech (vector or .wav file) is longer than 3 seconds
%multiple results may be produced. S specifies the number of samples to
%move ahead in the vector or file when extracting the next segment.
%The default value of 48,000 gives zero overlap between
%segments. Using this default any input less than 6 sec. will produce one
%result, based on just the first 3 sec. A 6 sec. input will produce two
%results. If S = 24,000 for example, segment overlap will be 50%, a 
%4.5 sec. input will produce 2 results and a 6 sec. input will produce
%3 results. In all cases an additional final value is produced. That 
%value is the mean of all segment results that are associated with 
%segments that have speech activity factor greater than 0.45. If there are
%no such segments, the final result is NaN.
%
%ctlInfo.channel = C specifies a channel number to use when the input
%speech is in a multi-channel .wav file. Default is C = 1.
%
%ctlInfo.outFilename = 'myFile.txt' specifies a text file that
%captures WAWEnet results on a new line for each speech input processed.
%If the file exists it will be appended. The extension .txt will be added
%as needed. Default is that no .txt file is generated.
%
%--------------------------------OUTPUTS--------------------------------
%The output for each of the N speech signals processed is in the 1 by N
%structure allFileInfo. The fields are:
%allFileInfo.name, .wav filename or "MatlabVector"
%allFileInfo.exception, description of any exception encountered
%allFileInfo.sampleRate, native sample rate of the signal
%allFileInfo.duration, duration of the signal in seconds
%allFileInfo.activeLevel, active speech level of final segment 
%   in dB below overload
%allFileInfo.activityFactor, speech activity factor of final segment
%allfileInfo.allActivityFactors, speech activity factors for each segment
%allfileInfo.allActiveLevels, active speech level for each segment
%allFileInfo.netOut, output value(s) produced by WAWEnet.  Contains one
%value for each segment processed, plus a final value that is the mean
%over all segments that have speech activity factor greater than 0.45.
%
%ctlInfo is returned as well.  In addition to the five fields defined
%in the input section above, it also includes:
%ctlInfo.sampleRate, native sample rate for WAWEnets, always 16,000 smp/s
%ctlInfo.segmentLength, native segment length rate for WAWEnet,
%   always 48,000 smp (3 s)
%ctlInfo.outFileHeader, the header that defines results shown on screen
%   and in any output file.  This includes a starting time/date stamp.
%
%References:
%[1] A. A. Catellier and S.D. Voran, ``WAWEnets: A No-Reference 
%Convolutional Waveform-Based Approach to Estimating Narrowband and 
%Wideband Speech Quality,'' in Proc. 45th IEEE International Conference
%on Acoustics, Speech and Signal Processing, May 2020.
%[2] ITU-T Recommendation P.862, ``Perceptual evaluation of speech quality
%(PESQ),'' Geneva, 2001.
%[3] ITU-T Recommendation P.863, ``Perceptual objective listening quality
%analysis,'' Geneva, 2018.
%[4] R. Huber and B. Kollmeier, ``PEMO-Q - A new method for objective audio
%quality assessment using a model of auditory perception,'' IEEE Trans.
%ASLP, vol. 14, no. 6, pp. 1902-1911, Nov. 2006.
%[5] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, ``An algorithm
%for intelligibility prediction of time�frequency weighted noisy speech,''
%IEEE Trans. ASLP, vol. 19, no. 7, pp. 2125-2136, Sep. 2011.

activityThreshold = 0.45; %only speech segements that exceed this 
%activity threshold are included in grand average result

if isempty(varargin) %check for optional input and assign it
    ctlInfo = [];
else
    ctlInfo =varargin{1};
end

ctlInfo = setDefaults(ctlInfo); %set defaults where not specified by user

if isnumeric(inSpeech) %input is vector of audio samples
    %initial processing of audio samples
    [audioSamples, fileInfo] = processInSpeech(inSpeech,ctlInfo);
    nAudioFiles = 1; %only 1 signal to process
    fileMode = 0; %loading .wav files will not be necessary because
        %audio was passed in as Matlab vector
        
    %Produce warning if sample magnitude exceeds 1.0. This could be OK.
    %For example it could be caused by a processing chain, even when
    %nominal signal range started as at +/- 1. But it could be not OK. For
    %example when it is caused by loading in audio with a range of
    %+/- 2^15 and failing to rescale it.
    if 1 < max(abs(audioSamples))
        warning( ...
        'Input audio vector sample values outside nominal range (+/- 1.0)')
    end
else    %input is a filename
    infileList = processInFile(inSpeech); %create list of .wav files
    nAudioFiles = length(infileList); %number of files in list
    fileMode = 1; %loading .wav files is necessary
end

%load WAWEnet parameters for selected mode
modeString = num2str(ctlInfo.WAWEnetMode);
parsFileName = ['WAWEnetPars_',modeString,'.mat'];
WAWEnetPars = load(parsFileName);

ctlInfo.outFileHeader = makeHeader; %create text header
fprintf(ctlInfo.outFileHeader) %display header

%open output .txt file if needed
if ~isempty(ctlInfo.outFilename)
    outFID = fopen(ctlInfo.outFilename,'a+t');    
    fprintf(outFID,ctlInfo.outFileHeader); %write header to file 
end

for currentAudioFile = 1:nAudioFiles %loop over all files in list
    
    %if audio samples were not passed in as a vector then they must be
    %loaded from a .wav file
    if fileMode
        %extract current .wav filename from list
        fileInfo.name = infileList{currentAudioFile}; 
        %attempt to load contents of .wav file
        [fileInfo, audioSamples] = loadWaveFile(fileInfo,ctlInfo);
    end
    
    if isempty(fileInfo.exception) %valid speech input is available
        
        nAudioSamples = length(audioSamples); %find number of samples
        
        %find duration in seconds
        fileInfo.duration = nAudioSamples / ctlInfo.sampleRate ; 
        
        %find number of segments available
        nSegments = floor( ( nAudioSamples - ctlInfo.segmentLength ) ...
            / ctlInfo.segmentStep ) + 1;      
        
        netOut = zeros( 1, nSegments); %initialize network output variable
        
        %pointer to first audio sample of first segment
        firstSample = 1; 
        
        allActivityFactors=[]; %will save all speech activity factors
        allActiveLevels=[]; %will save all active speech levels 
        for currentSegment = 1:nSegments  %loop over all segments in file
            
            %calculate pointer to last audio sample of current segment
            lastSample = firstSample + ctlInfo.segmentLength - 1;
            
            %extract current segment
            currentAudioSamples = audioSamples(firstSample:lastSample);  
            
            %normalize current segment
            [currentAudioSamples, fileInfo] ...
                = audioNormalize(currentAudioSamples,ctlInfo,fileInfo);
            allActivityFactors(currentSegment) = fileInfo.activityFactor;
            allActiveLevels(currentSegment) = fileInfo.activeLevel;
            
            %apply WAWEnet to current segment
            netOut(currentSegment) = ...
                WAWEnetCNN(currentAudioSamples,WAWEnetPars);
            
            %update pointer to first audio sample of next segment 
            firstSample = firstSample + ctlInfo.segmentStep;          
            
        end %loop over all segments in file
        
        %average the outputs over all segments where speech activity factor
        %meets or exceeds threshold and append to the per segment values
        grandMean = mean(netOut(activityThreshold < allActivityFactors));
        if length(grandMean) == 0
          grandMean = NaN;
        end
        netOut = [netOut grandMean];
        fileInfo.netOut = netOut;
        fileInfo.allActivityFactors = allActivityFactors;
        fileInfo.allActiveLevels = allActiveLevels;

        
        %organize all results for reporting
        waveFileResults = createOutputString(fileInfo,ctlInfo);
        
    else %move exception message to the file results 
        waveFileResults = [fileInfo.name,fileInfo.exception];
    end
    
    fprintf(waveFileResults) %display current results
    
    %store current file results with others in output variable
    allFileInfo(currentAudioFile) = fileInfo;
    
    %write current results to .txt file if needed
    if ~isempty(ctlInfo.outFilename)
        fprintf(outFID,waveFileResults);  
    end
    
end %loop over all files in list

%close .txt file if needed
if ~isempty(ctlInfo.outFilename)
    fclose(outFID);  
end
%--------------------------------------------------------------------------

function [fileInfo, audioSamples] = loadWaveFile(fileInfo,ctlInfo)
%Loads specified channel of specified .wav file
%and converts sample rate to 16,000 smp/sec if needed.
%Will produce an exception message if file is not found, is 
%unreadable, or if any requirement on the file audio contents is unmet.

inputScaling = 32768/32767; %Scale factor needed to align this code
%with code that generated the CNN

fileInfo.exception = [];
fileInfo.sampleRate = [];
audioSamples = [];

%attempt to get .wav file info to test against requirements
try
    wavInfo = audioinfo(fileInfo.name); 
catch
    fileInfo.exception = ':  Not found or not a readable .wav file.\n';
    return
end

%test info against requirements for compression, channels, sample rate
%and duration
if strcmpi(wavInfo.CompressionMethod,'compressed')
    fileInfo.exception = ...
        ':  This .wav file is not uncompressed.\n';
elseif wavInfo.NumChannels < ctlInfo.channel
    fileInfo.exception = ...
        ':  This .wav file does not contain the requested channel.\n';
elseif ~any(wavInfo.SampleRate == [8000 16000 24000 32000 48000])
    fileInfo.exception = ...
        ':  This .wav file does have required sample rate (8, 16, 24, 32, or 48k).\n';
elseif ( wavInfo.TotalSamples / wavInfo.SampleRate ) < 3
    fileInfo.exception = ...
        ':  This  .wav file has duration less than 3 seconds.\n';
else %all requirements have been met, read the .wav contents
    [audioSamples, fileInfo.sampleRate] = audioread(fileInfo.name);
    audioSamples = audioSamples(:,ctlInfo.channel); %select channel
    
    %Input processing required to exactly replicate the audio input
    %processing in the code that generated the CNN
    audioSamples = max(-1,audioSamples*inputScaling); 
    
    signalCheck = sum(abs(audioSamples)); %enable test no signal
    if signalCheck == 0  %test signal validity
        fileInfo.exception = ':  This .wav file has no signal.\n';
    else %valid signal found, convert rate to 16,000 smp/sec if needed
        if fileInfo.sampleRate == 8000
            audioSamples = upSampleByTwo(audioSamples);
        elseif fileInfo.sampleRate == 24000
            audioSamples = upSampleByTwo(audioSamples);
            audioSamples = downSampleByThree(audioSamples);
        elseif fileInfo.sampleRate == 32000
            audioSamples = downSampleByTwo(audioSamples);
        elseif fileInfo.sampleRate == 48000
            audioSamples = downSampleByThree(audioSamples);
        end
    end
end
%--------------------------------------------------------------------------
            
function [outSamples, fileInfo] = ...
    audioNormalize(inSamples,ctlInfo,fileInfo)
%Removes mean of input signal.
%Measures active speech level and speech activity factor of input signal.
%Adjusts active speech level to -26 dB if levelNormalization is requested.

targetLevel = -26;
nSamples = length(inSamples); %number of samples in input signal

%calculate and remove mean from input signal
inSamples = inSamples - sum(inSamples)/nSamples;

%measure input signal
[fileInfo.activeLevel, fileInfo.activityFactor] = ...
    levelMeter(inSamples,ctlInfo);

if ctlInfo.levelNormalization
    %calculate gain required to reach target level
    gain = 10^((targetLevel - fileInfo.activeLevel) / 20); 
    outSamples = inSamples * gain; %apply calculated gain
else
    outSamples = inSamples;
end
%--------------------------------------------------------------------------

function [ activeSpeechLevel, speechActivityFactor ] = ...
    levelMeter( inSamples , ctlInfo )
%Calculates an active speech level in dB and speech activity
%factor (0 to 1) from an input signal.
%Sample rate (in smp/sec) is required to properly calculate speech
%envelope and apply hangover times.
%Based on ITU-T P.56 (12/2011) "Objective measurement of active 
%speech level"

%set filter coefficient for time constant of 30 ms
gFilter = exp(-1/(ctlInfo.sampleRate * 0.03)); 

sumSquares=sum(inSamples.^2); %sum of all samples squared
inSamples=abs(inSamples); %rectify input signal

%smooth rectified input signal to get speech envelope
inSamples = IIRfilter((1-gFilter)^2,[gFilter^2 -2*gFilter],inSamples);

%reduce envelope sample rate to 500 Hz (2 ms resolution)
downSampleFactor = round(ctlInfo.sampleRate/500);

inSamples = inSamples(1:downSampleFactor:end); %downsample
nSamples=length(inSamples); %number of samples in new envelope

%number of samples in 200 ms "hangover time"
hangSamples = round(.2 * 500);

%scale down to reflect downsampling
sumSquares = sumSquares / downSampleFactor;

logEnv = floor(log2(max(inSamples,2^-20))); %calculate log envelope
logEnv = min(logEnv,1); %upper limit on log envelope is 1.0

%Matrix will hold ones and zeros to indicate speech activity at 16 
%different levels 
activityMatrix = zeros(nSamples,16);

%Loop to mark activity at highest level and all lower levels
for sample=1:nSamples
    
    %will mark a block of samples, consistent with the hangover time
    lastSample = min( nSamples, sample + hangSamples - 1);
    
    %Mark activity at highest level and all lower levels
    %in current timeslot and following slots according to hangover time
    activityMatrix( sample:lastSample, 1:logEnv(sample) + 16) = 1;
end

const = 20*log10(2); %log base and dB conversion value needed in loop below
logActivity = zeros(16,1);
logDiff = zeros(16,1);

%loop to calculate log activity and log difference at all 16 levels
for i=1:16
    totalActive = sum(activityMatrix(:,i)); %total one's in current column
    if 0 < totalActive
        logActivity(i) = 10*log10(sumSquares / totalActive);  
        logDiff(i) = logActivity(i) - (i-16) * const;
    else
        logActivity(i) = 100;  
        logDiff(i) = 100;    
    end    
end

activeSpeechLevel = -100; %value returned if there is no activity

%loop over levels to find if and where logDiff crosses 15.9 dB threshold
for i=1:15
   if logDiff(i) >= 15.9 && logDiff(i+1) <= 15.9
      if logDiff(i) == logDiff(i+1)
         activeSpeechLevel = logActivity(i);
      else
         %perform linear interpolation
         %find the proper scaling of the step in logDiff needed to hit
         %15.9 dB threshold 
         scaledStep = ( logDiff(i)-15.9 ) / ( logDiff(i)-logDiff(i+1));
         
         %apply that same scaling to logActivity
         activeSpeechLevel = (1-scaledStep) * logActivity(i) ...
             + scaledStep * logActivity(i+1);
      end 
   end
end

%calculate speech activity factor
if -100 < activeSpeechLevel
    speechActivityFactor = (sumSquares / nSamples) * ...
        (10^ (- activeSpeechLevel/10));
else
    speechActivityFactor = 0;
end
%--------------------------------------------------------------------------

function outSignal=IIRfilter(bCoeff,aCoeffs,inSignal)
%Implements a specific IIR filter in direct form:
%y(n) = b(1)*x(n) - a(1)*y(n-2) - a(2)*y(n-1)
%(where x is input sequence and y is output sequence)
%inSignal is a column vector of input samples
%outSignal is a column vector of output samples
%bCoeff is a single MA filter coefficient, defined in equation above
%aCoeffs is a row vector containing two AR filter coefficients, defined 
%in equation above

nIn = length(inSignal); %find number of samples in input signal
inSignal = [0;0;inSignal]; %zero pad with two zeros at front of signal
outSignal=zeros(nIn+2,1); %initialize output signal

%Loop over input
for i=3:nIn+2
    %second term is inner product between row vector and column vector
    %it represents two mults and one add
    outSignal(i)=inSignal(i) * bCoeff - aCoeffs * outSignal(i-2:i-1);
end 

%Retain relevant portion of output
outSignal=outSignal(3:end);
%--------------------------------------------------------------------------

function outString = createOutputString(fileInfo,ctlInfo)
%Extracts needed results from two structures and build a string of results
%for output to screen (and file).

%Replace any backslashes in filename so fprintf will display them
escFilename = fileInfo.name;
%`replace` not implemented in octave yet.
escFilename(fileInfo.name == '\') = '/';
outString = [escFilename,':'];
outString = [outString,' ',num2str(ctlInfo.channel)];
outString = [outString,' ',num2str(fileInfo.sampleRate)];
outString = [outString,' ',num2str(fileInfo.duration)];
outString = [outString,' ',num2str(fileInfo.activeLevel)];
outString = [outString,' ',num2str(fileInfo.activityFactor)];
outString = [outString,' ',num2str(ctlInfo.levelNormalization)];
outString = [outString,' ',num2str(ctlInfo.segmentStep)];
outString = [outString,' ',num2str(ctlInfo.WAWEnetMode)];
if length(fileInfo.netOut) > 1
  outString = [outString, ' [', num2str(fileInfo.netOut(1:end - 1)), ']'];
  outString = [outString, ' ', num2str(fileInfo.netOut(end)),'\n'];
else
  outString = [outString,' ',num2str(fileInfo.netOut),'\n'];
end
%--------------------------------------------------------------------------

function outString = makeHeader
%Create text header that identifies results
outString = ['\n'];
outString = [outString,datestr(now),'\n'];
outString = [outString,'filename, channel, native sample rate, duration (sec),\n'];
outString = [outString,'  active speech level (dB), speech activity factor,\n'];
outString = [outString,'    level norm., segment step (smp), WAWEnet mode, '];
outString = [outString ,'WAWEnet output(s)\n'];
outString = [outString ,'--------------------------------'];
outString = [outString ,'----------------------------------------\n'];
%--------------------------------------------------------------------------

function ctlInfo = setDefaults(ctlInfo)
%Vets input and sets default values as appropriate for 7 fields of ctlInfo:
%ctlInfo.sampleRate always set to 16000
%ctlInfo.segmentLength always set to 48000
%ctlInfo.WAWEnetMode defaults to 1,
%   valid user inputs are 1,2,3,4
%ctlInfo.levelNormalization defaults to 1,
%   value user inputs are 0,1
%ctlInfo.segmentStep defaults to 48000,
%   valid user inputs are integers 1 and greater 
%ctlInfo.channel defaults to 1,
%   valid user inputs are integers 1 and greater
%ctlInfo.outFilename defaults to empty variable, 
%   user input is unconstrained, .txt is added as needed 

ctlInfo.sampleRate = 16000; %(smp/sec) rate that WAWEnets use
ctlInfo.segmentLength=48000; %16,000 smp/sec x 3 seconds

if isfield(ctlInfo,'WAWEnetMode') %if field exists
     if ~any(ctlInfo.WAWEnetMode == [1 2 3 4]) %test for valid value
         disp('Invalid value of ctlInfo.WAWEnetMode ignored and')
         disp('default value (1) used.')
         ctlInfo.WAWEnetMode = 1;
     end    
else
    ctlInfo.WAWEnetMode = 1; %field does not exist, set to default
end

if isfield(ctlInfo,'levelNormalization') %if field exists
     if ~any(ctlInfo.levelNormalization == [0 1]) %test for valid value
         disp('Invalid value of ctlInfo.levelNormalization ignored and')
         disp('default value (1) used.')
         ctlInfo.levelNormalization = 1;
     end    
else
    ctlInfo.levelNormalization = 1; %field does not exist, set to default
end

if isfield(ctlInfo,'segmentStep') %if field exists
    %test for an integer one or greater
     if ~(ctlInfo.segmentStep == floor(ctlInfo.segmentStep) ...
                 && 1 <= ctlInfo.segmentStep) 
         disp('Invalid value of ctlInfo.segmentStep ignored and')
         disp('default value (48000) used.')
         ctlInfo.segmentStep = 48000;
     end    
else
    ctlInfo.segmentStep = 48000; %field does not exist, set to default
end

if isfield(ctlInfo,'channel') %if field exists
     %test for an integer one or greater
     if ~(ctlInfo.channel == floor(ctlInfo.channel) ...
                 && 1 <= ctlInfo.channel)
         disp('Invalid value of ctlInfo.channel ignored and')
         disp('default value (1) used.')
         ctlInfo.channel = 1;
     end    
else
    ctlInfo.channel = 1; %field does not exist, set to default
end

if isfield(ctlInfo,'outFilename') %if field exists
     inputName = ctlInfo.outFilename; %extract the name
     if 5 <= length(inputName) %if name has at least 5 characters
         %test to see if final 4 are '.txt'
         if strcmpi(inputName(end-3:end),'.txt')
             %if yes, use the specified name
             ctlInfo.outFilename = inputName; 
         else %otherwise append '.txt'       
             ctlInfo.outFilename = [inputName,'.txt']; 
         end       
     elseif 0 < length(inputName) %fewer than 5 but not empty
        ctlInfo.outFilename = [inputName,'.txt'];       
     end     
else
    ctlInfo.outFilename = []; %field does not exist, set to default
end
%--------------------------------------------------------------------------

function [audioSamples, fileInfo] = processInSpeech(inSpeech,ctlInfo)
%Checks that vector of input speech samples is valid and forces it to be
%a column vector.  Adds appropriate info to fileInfo.

inputScaling = 32768/32767; %Scale factor needed to align this code
%with code that generated the CNN
    
fileInfo.exception = [];
fileInfo.sampleRate = 16000;
fileInfo.name = 'MatlabVector'; 
audioSamples = [];
    
%if multiple columns, transpose in attempt to get column vector 
if 1 <  size(inSpeech,2)
    inSpeech = inSpeech';
end

%if still multiple columns, input was matrix, not vector
if 1 <  size(inSpeech,2)
     fileInfo.exception = ...
        ':  Input variable is not a row or column vector.\n';
elseif size(inSpeech,1) < ctlInfo.segmentLength %too short
    fileInfo.exception = ...
        ':  Input variable has duration less than 3 seconds.\n';
elseif 0 == sum(abs(inSpeech)) %no signal
    fileInfo.exception = ':  Input variable has no signal.\n';
else %passes all tests
    audioSamples = inSpeech;
    
    %Input processing required to exactly replicate the audio input
    %processing in the code that generated the CNN.
    audioSamples = max(-1,audioSamples*inputScaling);  
end
%--------------------------------------------------------------------------

function infileList = processInFile(infile)
%Process a filename that may be .wav or .txt.
%If .wav that one name forms the entire list that is returned in cell
%array.
%If .txt the contents are read and they become the list that is returned
%in cell array.

if length(infile)<4
    error('infile must have extension .wav or .txt')
end

%if it is .wav, then it is the entire list of .wav files
if strcmpi(infile(end-3:end),'.wav')
    infileList = {infile}; 
    
%if it is .txt, read contents in to form the list    
elseif strcmpi(infile(end-3:end),'.txt')
    infileList = importdata(infile);       
else
    error('infile must have extension .wav or .txt')
end
%--------------------------------------------------------------------------

function outSamples = upSampleByTwo(inSamples)
%Doubles sample rate by inserting a zero after each original sample, then
%low-pass filtering that signal at half of the new Nyquist frequency.
%inSamples and outSamples are column vectors
%outSamples has exactly twice as many samples as inSamples

%59 filter coefficients needed for up or down sampling by factor of 2.
%Source is ITU-T G.191 (1/2019) "Software tools for speech and audio
%coding standardization," available at itu.ch. More specifically, these
%coefficients are found in fill_lp_2_to_1() inside of in fir-flat.c.
coeffs = [1584 805 -4192 -8985 -5987 2583 4657 -3035 -7004 1542 8969 ...
    567 -10924 -3757 12320 7951 -12793 -13048 11923 18793 -9331 -24802 ...
    4694 30570 2233 -35439 -11526 38680 23114 -39474 -36701 36999 ...
    51797 -30419 -67658 18962 83318 -1927 -97566 -21284 108971 51215 ...
    -115837 -88430 116130 133716 -107253 -188497 85497 255795 -44643 ...
    -342699  -28185 468096 167799 -696809 -519818 1446093 3562497];

coeffs=[coeffs,fliplr(coeffs)]; %Extend symmetrically to get length 118

coeffs=coeffs/4179847; %normalize so sum is 2.0 to get gain of 1.0 at DC

nSamples = length(inSamples); %number of input samples

%Temporary signal will have twice the samples of the input, plus
%59 zeros at start and 58 zeros at end to get zero delay
tempSamples = zeros(59 + 2*nSamples + 58,1);

%Place original samples at every other location
%(after initial 59 zeros)
for samplePointer = 1 : nSamples
    tempSamples(59 + 2*samplePointer - 1) = inSamples(samplePointer);
end

%Initialize output column vector
outSamples = zeros(2*nSamples,1);

%Low-pass filter the temporary signal to produce the output signal
for samplePointer = 1 : 2*nSamples
    %This is an inner product between a row vector and a column vector
    %it includes 118 mults and 117 adds
    outSamples(samplePointer) = ...
        coeffs*tempSamples(samplePointer:samplePointer+117);
end
%--------------------------------------------------------------------------

function outSamples = downSampleByTwo(inSamples)
%Halves the sample rate by low-pass filtering the input signal at half of
%the Nyquist frequency, then deleting every other sample.
%
%inSamples and outSamples are column vectors
%If the number of samples in inSamples is nSamples, then the number of 
%samples in outSamples is ceil(nSamples/2)

%59 filter coefficients needed for up or down sampling by factor of 2.
%Source is ITU-T G.191 (1/2019) "Software tools for speech and audio
%coding standardization," available at itu.ch. More specifically, these
%coefficients are found in fill_lp_2_to_1() inside of in fir-flat.c.
coeffs = [1584 805 -4192 -8985 -5987 2583 4657 -3035 -7004 1542 8969 ...
    567 -10924 -3757 12320 7951 -12793 -13048 11923 18793 -9331 -24802 ...
    4694 30570 2233 -35439 -11526 38680 23114 -39474 -36701 36999 ...
    51797 -30419 -67658 18962 83318 -1927 -97566 -21284 108971 51215 ...
    -115837 -88430 116130 133716 -107253 -188497 85497 255795 -44643 ...
    -342699  -28185 468096 167799 -696809 -519818 1446093 3562497];

coeffs=[coeffs,fliplr(coeffs)]; %Extend symmetrically to get length 118

coeffs=coeffs/8359694; %normalize so sum is 1.0 to get gain of 1.0 at DC

nSamples = length(inSamples); %number of input samples

%add 59 zeros at start and 58 zeros at end to get zero delay
inSamples = [zeros(59,1);inSamples;zeros(58,1)];

%Initialize output column vector
outSamples = zeros(nSamples,1);

%low-pass filter this zero-padded signal
for samplePointer = 1 : nSamples
    %This is an inner product between a row vector and a column vector
    %it includes 118 mults and 117 adds
    outSamples(samplePointer) = ...
        coeffs*inSamples(samplePointer:samplePointer+117);
end
outSamples = outSamples(1:2:end); %retain every other sample.
%--------------------------------------------------------------------------

function outSamples = downSampleByThree(inSamples)
%Reduces sample rate by a factor of three.  Accomplished by low-pass
%filtering the input signal at one-third of the Nyquist frequency, then
%deleting two samples after each retained sample.
%
%inSamples and outSamples are column vectors
%If the number of samples in inSamples is nSamples, then the number of 
%samples in outSamples is ceil(nSamples/3)

%84 filter coefficients needed for up or down sampling by factor of 3.
%Source is ITU-T G.191 (1/2019) "Software tools for speech and audio
%coding standardization," available at itu.ch. More specifically, these
%coefficients are found in fill_lp_3_to_1() inside of in fir-flat.c.
coeffs=[877 3745 6479 8447 7307 3099 -2223 -5302 -3991 766 5168 5362 ...
    731 -5140 -7094 -2830 4611 8861 5584 -3260 -10326 -8887 888 11145 ...
    12532 2617 -10961  -16207 -7257 9442 19522 12931 -6288 -22007 ...
    -19398 1280 23148 26290 5704 -22403 -33102 -14655 19237 39196 ...
    25404 -13162 -43824 -37610 3766 46146 50752 9264 -45243 -64134 ...
    -26137 40124 76873 46957 -29705 -87899 -71773 12729 95920 100661 ...
    12412 -99329 -133927 -48113 95967 172563 98654 -82409 -219347 ...
    -173208 51783 282060 295863 14257 -387590 -556360 -195882 696028 ...
    1767624 2494432];

coeffs=[coeffs,fliplr(coeffs)]; %Extend symmetrically to get length 168

coeffs=coeffs/8436450; %normalize so sum is 1.0 to get gain of 1.0 at DC

nSamples = length(inSamples); %number of input samples

%add 83 zeros at start and 84 zeros at end to get zero delay
inSamples = [zeros(83,1);inSamples;zeros(84,1)];

%Initialize output column vector
outSamples = zeros(nSamples,1);

%low-pass filter this zero-padded signal
for samplePointer = 1:nSamples
    %This is an inner product between a row vector and a column vector
    %it includes 168 mults and 167 adds
    outSamples(samplePointer) = ...
        coeffs*inSamples(samplePointer:samplePointer+167);
end
outSamples = outSamples(1:3:end); %retain every third sample.
%--------------------------------------------------------------------------