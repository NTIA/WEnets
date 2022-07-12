#/bin/bash

echo "This file demos WAWEnet functionality and provides some verification that the code has been compiled correctly."
echo "Assumes ./WAWEnet is in this directory, and that sample speech is located in ../../speech"

read -p "Please enter the path to libtorch/lib on your system: " libtorch_path

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH=$libtorch_path
    echo $LD_LIBRARY_PATH
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH=$libtorch_path
    echo $DYLD_LIBRARY_PATH
else
    echo "demo.sh not implemented for this platform"
fi

echo "*****************************************************************"
# One .wav file
./WAWEnet ../../speech/T000439_Q231_D401.wav
echo "result should be: "
echo   "../../speech/T000439_Q231_D401.wav:  1 16000 3.000000 -25.978893 0.495506 1 48000 1 [ 1.569181 ] 1.569181"

echo "*****************************************************************"
# list of 10 .wav files, results to screen
./WAWEnet list.txt

# list of 10 .wav files, results to screen and myText.txt
./WAWEnet list.txt -o myText.txt

echo "the output of both commands should be:"
echo "../../speech/T000053_Q159_D401.wav:  1 16000 3.000000 -26.034792 0.915251 1 48000 1 [ 1.222245 ] 1.222245"
echo "../../speech/T000093_Q446_D401.wav:  1 16000 3.000000 -25.992599 0.498567 1 48000 1 [ 4.194879 ] 4.194879"
echo "../../speech/T000342_Q125_D401.wav:  1 16000 3.000000 -25.999733 0.988531 1 48000 1 [ 1.181551 ] 1.181551"
echo "../../speech/T000439_Q231_D401.wav:  1 16000 3.000000 -25.978893 0.495506 1 48000 1 [ 1.569181 ] 1.569181"
echo "../../speech/T000493_Q415_D401.wav:  1 16000 3.000000 -25.993883 0.747076 1 48000 1 [ 2.939228 ] 2.939228"
echo "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 1 [ 3.675040 ] 3.675040"
echo "../../speech/T000937_Q340_D401.wav:  1 16000 3.000000 -25.992653 0.793695 1 48000 1 [ 2.467979 ] 2.467979"
echo "../../speech/T001002_Q366_D401.wav:  1 16000 3.000000 -25.985020 0.820408 1 48000 1 [ 3.668306 ] 3.668306"
echo "../../speech/T001121_Q269_D401.wav:  1 16000 3.000000 -25.985125 0.572003 1 48000 1 [ 2.789083 ] 2.789083"
echo "../../speech/T001189_Q200_D401.wav:  1 16000 3.000000 -25.983643 0.491644 1 48000 1 [ 2.211183 ] 2.211183"
echo "../../speech/long.wav:  1 16000 4.717500 -24.886742 0.807339 1 48000 1 [ 4.229949 ] 4.229949"


echo "*****************************************************************"
# One .wav file, 4.7 sec long, results to screen, explicit selection of 4 inputs
#mode 2
#level normalization off
#segment step of 12000 gives 75% segment, overlap, will give 3 segments and 3 results
#channel 1 of .wav file explicitly selected
./WAWEnet ../../speech/long.wav -m 2 -l 0 -s 12000 -c 1
echo "result should be: "
echo "../../speech/long.wav:  1 16000 4.717500 -27.232590 0.925942 0 12000 2 [ 4.466176 4.156981 4.396053  ]  4.339736"


echo "*****************************************************************"
# One .wav file, ~10 sec long, with no speech, results to screen
./WAWEnet ../../speech/long_silence.wav -m 2 -l 0 -s 12000
echo "result should be: "
echo "../../speech/long_silence.wav:  1 16000 10.025187 -100.000000 0.000000 0 12000 2 [ 2.832274 2.831977 2.830791 2.833440 2.833960 2.830233 2.828940 2.832784 2.830861 2.832243  ]  nan"


echo "*****************************************************************"
# One .wav file with 2 channels.
#channel 1 of .wav file explicitly selected
#This channel is identical to ../speech/T000053_Q159_D401.wav
./WAWEnet ../../speech/TwoChannel_53_93.wav -c 1
echo "result should be: "
echo "../../speech/TwoChannel_53_93.wav:  1 16000 3.000000 -26.034792 0.915251 1 48000 1 [ 1.222245 ] 1.222245"


echo "*****************************************************************"
#channel 2 of .wav file explicitly selected
#This channel is identical to ../speech/T000093_Q446_D401.wav
./WAWEnet ../../speech/TwoChannel_53_93.wav -c 2
echo "result should be: "
echo "../../speech/TwoChannel_53_93.wav:  2 16000 3.000000 -25.992599 0.498567 1 48000 1 [ 4.194879 ] 4.194879"


echo "*****************************************************************"
# One .wav file, sample rate 8k results to screen only
./WAWEnet ../../speech/T000863_Q316_D401_8.wav
echo "result should be: "
echo "../../speech/T000863_Q316_D401_8.wav:  1 8000 3.000000 -25.597240 0.420938 1 48000 1 [ 2.835287 ] nan"


echo "*****************************************************************"
#One .wav file, sample rate 16k results to screen only
./WAWEnet ../../speech/T000863_Q316_D401.wav
echo "result should be: "
echo "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 1 [ 3.675040 ] 3.675040"


echo "*****************************************************************"
#One .wav file, sample rate 24k results to screen only
./WAWEnet ../../speech/T000863_Q316_D401_24.wav
echo "result should be: "
echo "../../speech/T000863_Q316_D401_24.wav:  1 24000 3.000000 -26.271122 0.517105 1 48000 1 [ 3.604044 ] 3.604044"


echo "*****************************************************************"
#One .wav file, sample rate 32k results to screen only
./WAWEnet ../../speech/T000863_Q316_D401_32.wav
echo "result should be: "
echo "../../speech/T000863_Q316_D401_32.wav:  1 32000 3.000000 -26.338856 0.530771 1 48000 1 [ 3.610513 ] 3.610513"


echo "*****************************************************************"
#One .wav file, sample rate 48k results to screen only
./WAWEnet ../../speech/T000863_Q316_D401_48.wav
echo "result should be: "
echo "../../speech/T000863_Q316_D401_48.wav:  1 48000 3.000000 -26.394070 0.522247 1 48000 1 [ 3.740452 ] 3.740452"


echo "*****************************************************************"
# One .wav file, mode 1
./WAWEnet ../../speech/T000863_Q316_D401.wav -m 1
echo "result should be: "
echo "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 1 [ 3.675040 ] 3.675040"


echo "*****************************************************************"
# One .wav file, mode 2
./WAWEnet ../../speech/T000863_Q316_D401.wav -m 2
echo "result should be: "
echo "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 2 [ 3.584861 ] 3.584861"


echo "*****************************************************************"
# One .wav file, mode 3
./WAWEnet ../../speech/T000863_Q316_D401.wav -m 3
echo "result should be: "
echo "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 3 [ 0.811744 ] 0.811744"


echo "*****************************************************************"
# One .wav file, mode 4
./WAWEnet ../../speech/T000863_Q316_D401.wav -m 4
echo "result should be: "
echo "../../speech/T000863_Q316_D401.wav:  1 16000 3.000000 -25.981779 0.480940 1 48000 4 [ 0.988499 ] 0.988499"

