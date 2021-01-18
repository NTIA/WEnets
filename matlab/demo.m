%% One .wav file, results to screen only
WAWEnet('../speech/T000439_Q231_D401.wav');
%produces:  ../speech/T000439_Q231_D401.wav: 1 16000 3 -25.9785 0.4955 1 48000 1 [1.5692] 1.5692

%% One .wav file, results to screen and Matlab structure
Results = WAWEnet('../speech/T000439_Q231_D401.wav');
%produces: ../speech/T000439_Q231_D401.wav: 1 16000 3 -25.9785 0.4955 1 48000 1 [1.5692] 1.5692

%% list of 10 .wav files, results to screen and Matlab structure
Results = WAWEnet('wavNames.txt');
% produces:
% ../speech/T000053_Q159_D401.wav: 1 16000 3 -26.0342 0.91525 1 48000 1 [1.2223] 1.2223
% ../speech/T000093_Q446_D401.wav: 1 16000 3 -25.9925 0.4986 1 48000 1 [4.1949] 4.1949
% ../speech/T000342_Q125_D401.wav: 1 16000 3 -25.9993 0.98853 1 48000 1 [1.1815] 1.1815
% ../speech/T000439_Q231_D401.wav: 1 16000 3 -25.9785 0.4955 1 48000 1 [1.5692] 1.5692
% ../speech/T000493_Q415_D401.wav: 1 16000 3 -25.9933 0.74707 1 48000 1 [2.9392] 2.9392
% ../speech/T000863_Q316_D401.wav: 1 16000 3 -25.9884 0.4817 1 48000 1 [3.6743] 3.6743
% ../speech/T000937_Q340_D401.wav: 1 16000 3 -26.0002 0.79517 1 48000 1 [2.4676] 2.4676
% ../speech/T001002_Q366_D401.wav: 1 16000 3 -25.9847 0.82047 1 48000 1 [3.6683] 3.6683
% ../speech/T001121_Q269_D401.wav: 1 16000 3 -25.9904 0.57274 1 48000 1 [2.7891] 2.7891
% ../speech/T001189_Q200_D401.wav: 1 16000 3 -25.9833 0.49164 1 48000 1 [2.2108] 2.2108

%% list of 10 .wav files, results to screen and myText.txt
clear myInfo
myInfo.outFilename = 'myText.txt';
WAWEnet('wavNames.txt',myInfo);
%produces:
% ../speech/T000053_Q159_D401.wav: 1 16000 3 -26.0342 0.91525 1 48000 1 [1.2223] 1.2223
% ../speech/T000093_Q446_D401.wav: 1 16000 3 -25.9925 0.4986 1 48000 1 [4.1949] 4.1949
% ../speech/T000342_Q125_D401.wav: 1 16000 3 -25.9993 0.98853 1 48000 1 [1.1815] 1.1815
% ../speech/T000439_Q231_D401.wav: 1 16000 3 -25.9785 0.4955 1 48000 1 [1.5692] 1.5692
% ../speech/T000493_Q415_D401.wav: 1 16000 3 -25.9933 0.74707 1 48000 1 [2.9392] 2.9392
% ../speech/T000863_Q316_D401.wav: 1 16000 3 -25.9884 0.4817 1 48000 1 [3.6743] 3.6743
% ../speech/T000937_Q340_D401.wav: 1 16000 3 -26.0002 0.79517 1 48000 1 [2.4676] 2.4676
% ../speech/T001002_Q366_D401.wav: 1 16000 3 -25.9847 0.82047 1 48000 1 [3.6683] 3.6683
% ../speech/T001121_Q269_D401.wav: 1 16000 3 -25.9904 0.57274 1 48000 1 [2.7891] 2.7891
% ../speech/T001189_Q200_D401.wav: 1 16000 3 -25.9833 0.49164 1 48000 1 [2.2108] 2.2108

%% Speech passed in via Matlab vector, results to screen and Matlab structure
speech = audioread('../speech/T000439_Q231_D401.wav');
Results = WAWEnet(speech);
%produces: MatlabVector: 1 16000 3 -25.9785 0.4955 1 48000 1 [1.5692] 1.5692

%% One .wav file, 4.7 sec long, results to screen, explicit selection of 
%4 inputs
clear myInfo
myInfo.WAWEnetMode = 2; %mode 2
myInfo.levelNormalization = 0; %level normalization off
myInfo.segmentStep = 12000;   %segment step of 12000 gives 75% segment,
%overlap, will give 3 segments and 3 results
myInfo.channel = 1; %channel 1 of .wav file explicitly selected
WAWEnet('../speech/long.wav',myInfo);
%produces: ../speech/long.wav: 1 16000 4.7175 -27.2326 0.92605 0 12000 2 [4.4661      4.1569       4.396] 4.3397

%% One .wav file, ~10 sec long, with no speech, results to screen, explicit
% selection of 3 inputs
clear myInfo
myInfo.WAWEnetMode = 2;
myInfo.levelNormalization = 0;
myInfo.segmentStep = 12000;
WAWEnet('../speech/long_silence.wav', myInfo);
%produces: ../speech/long_silence.wav: 1 16000 10.0252 -100 0 0 12000 2 [2.8323       2.832      2.8308      2.8335       2.834      2.8302       2.829      2.8328      2.8309      2.8323] NaN

%% One .wav file with 2 channels.
clear myInfo
myInfo.channel = 1; %channel 1 of .wav file explicitly selected
%This channel is identical to ../speech/T000053_Q159_D401.wav
WAWEnet('../speech/TwoChannel_53_93.wav',myInfo);
%produces: ../speech/TwoChannel_53_93.wav: 1 16000 3 -26.0342 0.91525 1 48000 1 [1.2223] 1.2223

myInfo.channel = 2; %channel 2 of .wav file explicitly selected
%This channel is identical to ../speech/T000093_Q446_D401.wav
WAWEnet('../speech/TwoChannel_53_93.wav',myInfo);
%produces: ../speech/TwoChannel_53_93.wav: 2 16000 3 -25.9925 0.4986 1 48000 1 [4.1949] 4.1949

%% One .wav file, sample rate 8k results to screen only
WAWEnet('../speech/T000863_Q316_D401_8.wav');
%produces: ../speech/T000863_Q316_D401_8.wav: 1 8000 3 -26.2047 0.48418 1 48000 1 [2.8602] 2.8602

%One .wav file, sample rate 16k results to screen only
WAWEnet('../speech/T000863_Q316_D401.wav');
%produces: ../speech/T000863_Q316_D401.wav: 1 16000 3 -25.9884 0.4817 1 48000 1 [3.6743] 3.6743

%One .wav file, sample rate 24k results to screen only
WAWEnet('../speech/T000863_Q316_D401_24.wav');
%produces: ../speech/T000863_Q316_D401_24.wav: 1 24000 3 -25.959 0.48128 1 48000 1 [3.6176] 3.6176

%One .wav file, sample rate 32k results to screen only
WAWEnet('../speech/T000863_Q316_D401_32.wav');
%produces: ../speech/T000863_Q316_D401_32.wav: 1 32000 3 -25.9072 0.4806 1 48000 1 [3.6404] 3.6404

%One .wav file, sample rate 48k results to screen only
WAWEnet('../speech/T000863_Q316_D401_48.wav');
%produces: ../speech/T000863_Q316_D401_48.wav: 1 48000 3 -26.0435 0.48179 1 48000 1 [3.7486] 3.7486

%% One .wav file, mode 1
clear myInfo
myInfo.WAWEnetMode = 1; 
WAWEnet('../speech/T000863_Q316_D401.wav',myInfo);
%produces: ../speech/T000863_Q316_D401.wav: 1 16000 3 -25.9884 0.4817 1 48000 1 [3.6743] 3.6743

%% One .wav file, mode 2
clear myInfo
myInfo.WAWEnetMode = 2; 
WAWEnet('../speech/T000863_Q316_D401.wav',myInfo);
%produces: ../speech/T000863_Q316_D401.wav: 1 16000 3 -25.9884 0.4817 1 48000 2 [3.5834] 3.5834

%% One .wav file, mode 3
clear myInfo
myInfo.WAWEnetMode = 3; 
WAWEnet('../speech/T000863_Q316_D401.wav',myInfo);
%produces: ../speech/T000863_Q316_D401.wav: 1 16000 3 -25.9884 0.4817 1 48000 3 [0.81169] 0.81169

%% One .wav file, mode 4
clear myInfo
myInfo.WAWEnetMode = 4; 
WAWEnet('../speech/T000863_Q316_D401.wav',myInfo);
%produces: ../speech/T000863_Q316_D401.wav: 1 16000 3 -25.9884 0.4817 1 48000 4 [0.98851] 0.98851
