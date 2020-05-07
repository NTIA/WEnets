#ifndef __WAWEnet_
#define __WAWEnet_

#include <string>
#include <vector>

#include <iostream>

#include <cmath>

#include <numeric>


#include <algorithm>

#include "AudioFile.h";

#include <windows.h>

using namespace std;



class ctlInfo
{
public:
    int sampleRate;
    int segLength;
    int segStep;
    int channel;
    int WAWEnetMode;
    string outFileHeader;
    string outFileName;
    int levelNormalization;
};
class FileInfo {
public:
    string name;
    string exception;
    float duration;
    vector<float> netOut;
    int sampleRate;
    float activeLevel;
    float speechActivityFactor;

};
class AudioInfo {
public:
    string CompressionMethod;
    int numChannels;
    int SampleRate;
    int totalSamples;
    float duration;
    string title;
    string comment;
    string artist;
    int bitsPerSample;

};
class Speech
{
public:
    float activeSpeechLevel;
    float speechActivityFactor;
};
class loadWaveFileInputs {
public:
    FileInfo fileInfo;
    vector<float> audiosamples;
    ctlInfo ctl;
};
class audioReadOutput {
public:
    vector<float> audiosamples;
    int SampleRate;
};
class audioNormalizeOutput {
public:
    vector<float> outSamples;
    FileInfo fileinfo;
};
/*


helper functions

*/
template<typename T>
vector<T> slice(vector<T>  v, int m, int n);

template<typename T>
vector<T> vectorAbs(vector<T>  vect);

template<typename T>
vector<T> removeMean(vector<T> vect, float sum);

template<typename T>
vector<T> multiVect(vector<T> vect, float multi);

template<typename T>
T dotMulti(vector<T> vect, vector<T>  vect2);

template<typename T>
T sumSquares(vector<T>  vect);

template<typename T>
vector<T> multiSlice(vector<T>  vect, float val);

template<typename T>
vector<T> MaxOrMin(vector<T> vect, float val, bool arg);

template<typename T>
vector<T> log2Vect(vector<T>  vect);

template<typename T>
vector<T> floorVect(vector<T>  vect);

template<typename T>
vector<T> flip(vector<T>  vect);

template<typename T>
T sumCol(vector<vector<T>> vect, int col);


vector<float> getZeroVector(float size);


vector<string> getInputPair(vector<string> arr, string option);

template<typename T>
T sumVector(vector<T> vect);

bool dirExists(const std::string& dirName_in);




/*


end of helper functions



*/

/*




start of main functions






*/






vector<float> upSampleByTwo(vector<float> inSamples);
vector<float> downSampleByTwo(vector<float> inSamples);
vector<float> downSampleByThree(vector<float> inSamples);

string makeHeader();

string createOutputString(FileInfo fileInfo, ctlInfo ctl);

ctlInfo parseVars(vector<string> vars, ctlInfo ctl);
vector<string> processInFile(string fileName);
AudioInfo getAudioInfo(FileInfo fileInfo);

audioReadOutput audioread(string fileName, ctlInfo ctl);

loadWaveFileInputs loadWaweFile(ctlInfo ctl, FileInfo fileInfo);

vector<float> IIRfilter(float bCoeff, vector<float> aCoeffs, vector<float> inS);

Speech levelMeter(vector<float> inSamplesA, ctlInfo ctl);

audioNormalizeOutput audioNormalize(vector<float> cAudio, ctlInfo ctl, FileInfo fInfo);

void WAWEnet(vector<string> fileAndArgs);


/*




end of main functions






*/
#endif