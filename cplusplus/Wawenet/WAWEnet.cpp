#include <sys/types.h>
#include <sys/stat.h>

#include "WAWEnetCNN.h"

#include "WAWEnet.h"


//Used to use cout for file writing to output file
#define _CRT_SECURE_NO_WARNINGS


/*
Start of helper functions for misc vector operations
*/

//gets a "slice" of a 1d vector with two paramters m and n
//ex: [1  2  3  4  5  6  7 ] , start = 1 , end = 4 -> [ 2  3  4  5 ]
template<typename T>
vector<T> slice(vector<T> v, int start, int end) {
    auto first = v.cbegin() + start;
    auto last = v.cbegin() + end;
    vector<T> vec(first, last);
    return vec;
}

//gets the absolute value of every element of a 1d vector
template<typename T>
vector<T> vectorAbs(vector<T> vect) {
    vector<T> retVect;
    for (int i = 0; i < vect.size(); ++i) {
        retVect.push_back(abs((float)vect.at(i)));
    }
    return retVect;
}

//remove the mean from the vector 
template<typename T>
vector<T> removeMean(vector<T>  vect, float sum) {
    vector<T> retVect = vect;
    for (int i = 0; i < vect.size(); ++i) {
        retVect.at(i) = retVect.at(i) - sum;
    }
    return retVect;
}

//multiply each element of the vector by multi
template<typename T>
vector<T> multiVect(vector<T>  vect, float multi) {
    vector<T> retVect;
    for (int i = 0; i < vect.size(); ++i) {
        retVect.push_back(vect.at(i)* multi);
    }
    return retVect;
}

//get the dot product of two vectors
template<typename T>
T dotMulti(vector<T> vect, vector<T> vect2) {
    T retVal = 0;
    if (vect.size() == vect2.size()) {

      for (int i = 0; i < vect.size(); ++i) {
        retVal += vect.at(i) * vect2.at(i);
      }
    }
    return retVal;
}

// get the all elements squares and added up
template<typename T>
T sumSquares(vector<T>  vect) {
    float retVal = 0;
    for (int i = 0; i < vect.size(); ++i) {
        retVal += pow(vect.at(i), 2);
    }
    return retVal;
}

//sum all of the elements of the vector to a single number
template<typename T>
T sumVector(vector<T>  vect) {
    float retVal = 0;
    for (int i = 0; i < vect.size(); ++i) {
        retVal += vect.at(i);
    }
    return retVal;
}

// calculate the mean of a vector
template<typename T>
T meanVector(vector<T> vect) {
    int vectorLength = vect.size();
    float vectorMean;
    if (vectorLength < 1) {
        vectorMean = nanf("");
    }
    else {
        float sum = 0.0;
        sum = sumVector(vect);
        vectorMean = sum / vectorLength;
    }
    return vectorMean;
}

//get a vector from every other nth element with n being val
template<typename T>
vector<T> multiSlice(vector<T>  vect, float val) {
    vector<float> retVect;
    for (int i = 0; i < vect.size(); i += val) {
        retVect.push_back(vect.at(i));
    }
    return retVect;
}

//arguments 1 is max, 0 is min
//sets the max or min of every element of the vector
template<typename T>
vector<T> MaxOrMin(vector<T> vect, float val,bool arg) {
    vector<float> retVect = vect;
    if (arg == true) {
        for (int i = 0; i < vect.size(); i++) {
            if (val > retVect.at(i)) {
                retVect.at(i) = val;
            }
        }
    }
    else {
        for (int i = 0; i < vect.size(); i++) { 
            if (val < retVect.at(i)) {
                retVect.at(i) = val;
            }
        }
    }
    return retVect;
}
//gets the log of every element of a vector
template<typename T>
vector<T> log2Vect(vector<T> vect) {
    vector<float> retVect = vect;
    for (int i = 0; i < vect.size(); i++) {
        retVect.at(i) = log2(retVect.at(i));
    }
    return retVect;
}

//gets the floor of every element of a vector
template<typename T>
vector<T> floorVect(vector<T> vect) {
    vector<float> retVect = vect;
    for (int i = 0; i < vect.size(); i++) {
        retVect.at(i) = floor(retVect.at(i));
    }
    return retVect;
}

//flips a 1d vector ex : [1  2  3] -> [3  2  1]

template<typename T>
vector<T> flip(vector<T>  vect) {
    vector<float> retVect = vect;
    reverse(retVect.begin(), retVect.end());
    return retVect;
}

//initializes a vector filled with zeros of with a inputted size

vector<float> getZeroVector(float size) {
    vector<float> zeroVect;

    zeroVect.assign(size, 0);
    return zeroVect;
}


//sums up only the column of a 2d vector

template<typename T>
T sumCol(vector<vector<T>> vect, int col) {
    float retVal = 0;

    for (int i = 0; i < vect.size(); ++i) {
        retVal += (vect.at(i)).at(col);
    }
    return retVal;
}

//based on a string option get a 1x2 vector for the $paramerter + associated input


vector<string> getInputPair(vector<string> arr,string option) {
    vector<string> output;
    auto strIt = find(arr.begin(), arr.end(), option);
    if (strIt != arr.end()) {
        int index = distance(arr.begin(), strIt);
        output.push_back(arr.at(index));
        index += 1;
        output.push_back(arr.at(index ));
    }
    return output;
}

//determines if the windows directory exists
bool dirExists(const std::string& dirName_in)
{
    struct stat info;
    return stat( dirName_in.c_str(), &info ) == 0 && info.st_mode & S_IFDIR;
}


/*
End of helper functions
*/


/*
beginning of main functions
*/


/*
Scaling functions so that the samples are at 16k

UpSampleByTwo  8k to 16k
*/
vector<float> upSampleByTwo(vector<float> inSamples) {

    vector<float> inSamplesE;
    inSamplesE = inSamples;

    vector<float> flipped;
    int nSamples;
    vector<float> tempSamples;
    vector<float> coeffs;
    vector<float> outSamples;



    // 59 filter coefficients needed for up or down sampling by factor of 2.
    // Source is ITU - T G.191 (1 / 2019) "Software tools for speech and audio
    // coding standardization, " available at itu.ch. More specifically, these
    // coefficients are found in fill_lp_2_to_1() inside of in fir-flat.c.

     coeffs = { 1584, 805 ,-4192 ,-8985, -5987 ,2583 ,4657,
        -3035, -7004 ,1542 ,8969 ,567, -10924, -3757, 12320 ,7951 ,-12793,
        -13048, 11923, 18793, -9331, -24802, 4694, 30570, 2233, -35439,
        -11526 ,38680 ,23114 ,-39474 ,-36701, 36999 ,51797, -30419, -67658 ,
        18962, 83318, -1927 ,-97566 ,-21284 ,108971 ,51215, -115837, -88430,
        116130, 133716 ,-107253, -188497 ,85497 ,255795 ,-44643 ,-342699,
        -28185, 468096, 167799, -696809, -519818, 1446093, 3562497 };

    //appends a reverse of the list to the end of the original list
    flipped = flip(coeffs);
    coeffs.insert(end(coeffs), begin(flipped), end(flipped));

    double div;
    //normalize for a gain of 1.0
    for (int i = 0; i < coeffs.size(); ++i) {
        //normalize so sum is 1.0 to get gain of 1.0 at DC
        div = coeffs.at(i) / 4179847;
        coeffs.at(i) = (float)div;
    }
    nSamples = inSamplesE.size();

    //Temporary signal will have twice the samples of the input, plus
    //59 zeros at start and 58 zeros at end to get zero delay
    tempSamples = getZeroVector(59 + (float)nSamples * 2 + 58);

    int val;

    for (int samplePointer = 0; samplePointer < nSamples; ++samplePointer) {
         val = 59 + 2 * samplePointer; //-1
        tempSamples.at(val) = inSamplesE.at(samplePointer);
    }


    outSamples = getZeroVector((float)2 * nSamples);



    vector<float> slicedVect;
    for (int sampleP = 0; sampleP < 2 * nSamples; ++sampleP) {
        //This is an inner product between a row vector and a column vector
        //it includes 118 mults and 117 adds
        slicedVect = slice(tempSamples, sampleP, sampleP + 117 + 1); // -1

        outSamples.at(sampleP) = dotMulti(coeffs, slicedVect);
    }
    return outSamples;


}


/*
 DownSampleByTwo  32k to 16k
*/
vector<float> downSampleByTwo(vector<float> inSamples) {

    vector<float> inSamplesE;
    inSamplesE = inSamples;
    vector<float> coeffs;
    vector<float> flipped;
    int nSamples;
    vector<float> slicedVect;
    vector<float> outSamples;

    // 59 filter coefficients needed for up or down sampling by factor of 2.
    // Source is ITU-T G.191 (1/2019) "Software tools for speech and audio
    // coding standardization," available at itu.ch. More specifically, these
    // coefficients are found in fill_lp_2_to_1() inside of in fir-flat.c.

    coeffs = { 1584, 805 ,-4192 ,-8985, -5987 ,2583 ,4657,
        -3035, -7004 ,1542 ,8969 ,567, -10924, -3757, 12320 ,7951 ,-12793,
        -13048, 11923, 18793, -9331, -24802, 4694, 30570, 2233, -35439,
        -11526 ,38680 ,23114 ,-39474 ,-36701, 36999 ,51797, -30419, -67658 ,
        18962, 83318, -1927 ,-97566 ,-21284 ,108971 ,51215, -115837, -88430,
        116130, 133716 ,-107253, -188497 ,85497 ,255795 ,-44643 ,-342699,
        -28185, 468096, 167799, -696809, -519818, 1446093, 3562497 };

    // appends a reverse of the list to the end of the original list
    flipped = flip(coeffs);
    coeffs.insert(end(coeffs), begin(flipped), end(flipped));

    double div;
    //normalize for a gain of 1.0
    for (int i = 0; i < coeffs.size(); ++i) {
        //normalize so sum is 1.0 to get gain of 1.0 at DC

        div = coeffs.at(i) / 8359694;
        coeffs.at(i) = (float)div;
    }
     nSamples = inSamplesE.size();

     //add 59 zeros to the beginning
     auto it = inSamplesE.insert(inSamplesE.begin(), 0);

     inSamplesE.insert(it, 58, 0);

     //add  58 zeros to the end of the vector for padding
     for (int i = 0; i < 58; ++i) {
         inSamplesE.push_back(0);
     }


    outSamples = getZeroVector((float)nSamples);

    for (int sampleP = 0; sampleP < nSamples; ++sampleP) {
        //This is an inner product between a row vector and a column vector
        //it includes 118 mults and 117 adds
        slicedVect = slice(inSamplesE, sampleP, sampleP + 117 + 1);
        outSamples.at(sampleP) = dotMulti(coeffs, slicedVect);
    }


    return  multiSlice(outSamples, 2);

}


/*
 DownSampleByThree 64k to 16k
*/
vector<float> downSampleByThree(vector<float> inSamples) {

    vector<float> inSamplesE;
    inSamplesE = inSamples;
    vector<float> coeffs;
    vector<float> flipped;
    int nSamples;
    vector<float> slicedVect;
    vector<float> outSamples;

    // 84 filter coefficients needed for up or down sampling by factor of 3.
    // Source is ITU-T G.191 (1/2019) "Software tools for speech and audio
    // coding standardization," available at itu.ch. More specifically, these
    // coefficients are found in fill_lp_3_to_1() inside of in fir-flat.c.

    coeffs = {
    877 ,3745 ,6479, 8447, 7307 ,3099 ,-2223 ,-5302 ,-3991 ,766 ,5168 ,5362,
    731 ,-5140 ,-7094 ,-2830 ,4611 ,8861 ,5584 ,-3260 ,-10326 ,-8887 ,888 ,11145,
    12532, 2617, -10961 ,-16207 ,-7257, 9442 ,19522 ,12931 ,-6288 ,-22007,
    -19398, 1280, 23148, 26290, 5704, -22403, -33102, -14655, 19237, 39196 ,
    25404, -13162 ,-43824, -37610, 3766 ,46146, 50752, 9264 ,-45243 ,-64134,
    -26137, 40124, 76873 ,46957 ,-29705, -87899, -71773, 12729, 95920 ,100661 ,
    12412, -99329, -133927, -48113, 95967, 172563, 98654, -82409, -219347,
    -173208, 51783, 282060, 295863, 14257, -387590, -556360 ,-195882, 696028,
    1767624, 2494432
    };

    // appends a reverse of the list to the end of the original list
    flipped = flip(coeffs);
    coeffs.insert(end(coeffs), begin(flipped), end(flipped));

    double div;
    //normalize for a gain of 1.0
    for (int i = 0; i < coeffs.size(); ++i) {
        //normalize so sum is 1.0 to get gain of 1.0 at DC

        div = coeffs.at(i) / 8436450;
        coeffs.at(i) = (float)div;
    }
    nSamples = inSamplesE.size();


    //add 83 to the beginning
    auto it = inSamplesE.insert(inSamplesE.begin(), 0);

    inSamplesE.insert(it, 82, 0);

    //add  84 zeros to the end of the vector for padding
    for (int i = 0; i < 84; ++i) {
        inSamplesE.push_back(0);
    }



    outSamples = getZeroVector((float)nSamples);



    for (int sampleP = 0; sampleP < nSamples; ++sampleP) {
        // This is an inner product between a row vector and a column vector
        // it includes 168 mults and 167 adds
        slicedVect = slice(inSamplesE, sampleP, sampleP + 167 + 1);
        outSamples.at(sampleP) = dotMulti(coeffs, slicedVect);
    }
    return multiSlice(outSamples, 3);

}

/*

 Makes the header for the output file

*/
string makeHeader() {
    //TODO add time?
    string header = "";
    header += "\n";
    header += "filename, channel, native sample rate, duration (sec),\n"
        "active speech level (dB), speech activity factor,\n"
        "level norm., segment step (smp), WAWEnet mode, "
        "WAWEnet output(s)\n"
        "--------------------------------"
        "----------------------------------------\n";
    return header;
}

/*
 Makes the body for the output file using fileInfo and ctlInfo 
*/

string createOutputString(FileInfo fileInfo, ctlInfo ctl) {


    string outString;
    FileInfo f;
    ctlInfo c;
    f = fileInfo;
    c = ctl;

    outString = "";
    outString += f.name;
    outString += ":  ";
    outString += std::to_string(c.channel);
    outString += " ";
    outString += std::to_string(c.sampleRate);
    outString += " ";
    outString += std::to_string(f.duration);
    outString += " ";
    outString += std::to_string(f.activeLevel);
    outString += " ";
    outString += std::to_string(f.speechActivityFactor);
    outString += " ";
    outString += std::to_string(c.levelNormalization);
    outString += " ";
    outString += std::to_string(c.segStep);
    outString += " ";
    outString += std::to_string(c.WAWEnetMode);
    outString += " ";
    if (f.netOut.size() > 1) {
        string array = "[ ";
        for (int i = 0; i < f.netOut.size(); ++i) {
            array += std::to_string(f.netOut.at(i));
            array += " ";
        }
        array += " ] ";
        outString += array;
    }
    else {
        outString += "[ ";
        outString += std::to_string(f.netOut.at(0));
        outString += " ]";
    }
    outString += " ";
    outString += std::to_string(f.grandMean);

    return outString;
}


/*

ParseVars simply parses the optional inputs on the command line as indicated below
 

Use WAWEnet(infile, '-m', M) to specify a WAWEnet mode.The integer M
 specifies the WAWEnet trained using a specific full - reference target.
 M = 1: WAWEnet trained using WB - PESQ[2] target values(Default)
 M = 2 : WAWEnet trained using POLQA[3] target values
 M = 3 : WAWEnet trained using PEMO[4] target values
 M = 4 : WAWEnet trained using STOI[5] target values

Use WAWEnet(infile, '-l', L) to specify internal level normalization of
 .wav file contents to 26 dB below clipping.
 L = 0: normalization off
 L = 1 : normalization on(Default)



Use WAWEnet(infile, '-s', S) to specify segment step(stride).
 WAWEnet requires a full 3 seconds of signal to generate a result.
 If a.wav file is longer than 3 seconds multiple results may be produced.
 S specifies the number of samples to move ahead in the file when
 extracting the next segment. The default value of S is 48, 000. This gives
 zero overlap between segments and any speech file less than 6 sec.
 will produce one result, based on just the first 3 seconds. 
 A 6 sec  speech file will produce two results.


 If S = 24, 000 for example, segment overlap will be 50 %, a 4.5 sec.
file will produce 2 results and a 6 sec file will produce 3 results.



Use WAWEnet(infile, '-c', C) to specify a channel number to use in the
 case of multi - channel.wav files.Default is C = 1.



Use WAWEnet(infile, '-o', 'myFile.txt') to specify a text file that
 captures WAWEnet results for each processed.wav file on a new line.
 If the file exists it will be appended.The extension.txt will be added
 as needed. Default is no.txt file generated.


*/

#define PATH_SEP "/"


ctlInfo parseVars(vector<string> vars, ctlInfo ctl) {
    int nArgs = vars.size();

    struct stat info;

    ctlInfo ctlI;
    ctlI = ctl;
    vector<string> output = vector<string>();
    vector<string> WaweMode = vector<string>();
    vector<string> SegmentStep = vector<string>();
    vector<string> channelP = vector<string>();
    vector<string> levelNorm = vector<string>();

    ctlI.outFileName = "";

     output = getInputPair(vars, "-o");
    if (!output.empty()) {
        string out = output.at(1);
        if (out.find_last_of(PATH_SEP) == string::npos) {
            string file = out;
            if (file.size() >= 5) {
                if (file.substr(file.size() - 4) == ".txt") {
                    ctlI.outFileName = file;
                }
                else {
                    ctlI.outFileName =  file + ".txt";

                }
            }
            else if (0 < file.size()) {
                ctlI.outFileName = file + ".txt";
            }
            else {
                cerr << "output file, " << file << " is invalid" << endl;
                throw std::runtime_error("output file is invalid");
            }
        }
        else {
            auto pathpos = out.find_last_of(PATH_SEP);
            string dir = out.substr(0, pathpos);
            string file = out.substr(pathpos + 1, out.length());
            if (dirExists(dir)) {
                if (file.size() >= 5) {
                    if (file.substr(file.size() - 4) == ".txt") {
                        ctlI.outFileName = dir + PATH_SEP + file;
                    }
                    else {
                        file += ".txt";
                        ctlI.outFileName = dir + PATH_SEP + file;

                    }
                }
                else if (0 < file.size()) {
                    file += ".txt";
                    ctlI.outFileName = dir + PATH_SEP + file;
                }
                else {
                    cerr << "output file, " << file << " is invalid" << endl;
                    throw std::runtime_error("output file is invalid");
                }
            }
            else {
                cerr << "directory " << dir << " does not exist" << endl;
                throw std::runtime_error("directory does not exist");

            }
        }
    }


    ctlI.WAWEnetMode = 1;

    const string WB_PESQ = "1";
    const string POLQA = "2";
    const string PEMO = "3";
    const string STOI = "4";


     WaweMode = getInputPair(vars, "-m");
    if (!WaweMode.empty()) {
        if (WaweMode.at(1) == WB_PESQ || WaweMode.at(1) == POLQA || WaweMode.at(1) == PEMO || WaweMode.at(1) == STOI) {
            
            ctlI.WAWEnetMode = stoi(WaweMode.at(1));
        }
        else {
            cerr << "improper number for wawemode " << stoi(WaweMode.at(1)) <<" , only modes 1-4 accepted" << endl;
            throw std::runtime_error("Improper number for wawemode");
        }

    }


    // specify default segment step: 48,000 samples = 3 seconds of 16k samp/sec speech
    ctlI.segStep = 48000;


    SegmentStep = getInputPair(vars, "-s");
    if (!SegmentStep.empty()) {
        auto seg = stof(SegmentStep.at(1));
        if (seg == floor(seg) && seg >= 1)
        {
            ctlI.segStep = seg;
        }
        else {
            cerr << "improper number for segStep " << stof(SegmentStep.at(1)) << endl;
            throw std::runtime_error("Improper number for segStep");
            
        }

    }


    ctlI.channel = 1;


     channelP = getInputPair(vars, "-c");
    if (!channelP.empty()) {
        float chan = stof(channelP.at(1));
        if (chan == floor(chan) && chan >= 1) {
            ctlI.channel = chan;
        }
        else {
            cerr << "improper number for channel " << stof(channelP.at(1))<< " channel must exist on current waveform" << endl;
            throw std::runtime_error("Improper number for channel");
            
        }
    }


    ctlI.levelNormalization = 1;


    levelNorm = getInputPair(vars, "-l");
    if (!levelNorm.empty()) {
        auto norm = stof(levelNorm.at(1));
        if (norm == 1 || norm == 0) {
            ctlI.levelNormalization = norm;
        }
        else {
            cerr << "improper number for level normalization " << stof(levelNorm.at(1)) <<
            "level normalization should be either 1(on) or 0(off)" << endl;
            throw std::runtime_error("Improper number for level Normalization");
            
        }

    }
    return ctlI;

}

/*

function to interpret the file to be a wav or a txt file
which are the two allowed inputs to the system
If it is a txt file then read all wav files listed and add
them to a list to be processed
If a wav file just add it to a 1x1 str array

*/

vector<string> processInFile(string fName) {
    string fileName;
    fileName = fName;

    vector<string> inFileList;

    string last4 = fileName.substr(fileName.size() - 4);
    if (last4 != ".wav" && last4 != ".txt") {
        cerr << "Is not a wav or txt file " << fileName.c_str() << endl;
        throw std::runtime_error("Invalid file type");
        return inFileList;
    }
    else if (last4 == ".wav") {
        inFileList.push_back(fileName);
        return inFileList;
    }
    else {
        string wavFiles;
        ifstream infile;
        string toVect;
        try {
            infile.open(fileName);
        }
        catch (std::ios_base::failure& e) {
            cerr << "txt file does not exist" << '\n';
            throw std::runtime_error("file not found exception");
        }

        while (getline(infile,wavFiles)) 
        {
            toVect = wavFiles;
            inFileList.push_back(toVect);
        }
        infile.close();

        if (inFileList.empty()) {
            cerr << "empty file list or file does not exist" << endl;
            throw std::runtime_error("empty file list");
        }

        return inFileList;

    }
}


/*

get the AudioInfo from the .wav file used when loading the file
set the audioInfo object and return from the .wav header
*/

AudioInfo getAudioInfo(FileInfo fileInfo) {


    AudioFile<float> audioFile;


    if (!audioFile.load(fileInfo.name)) {
        cerr << "file is not uncompressed" << endl;
        throw std::runtime_error("file is uncompressed");
    }

    audioFile.load(fileInfo.name);
    AudioInfo aInfo;

    aInfo.numChannels = audioFile.getNumChannels();
    aInfo.SampleRate = audioFile.getSampleRate();
    aInfo.bitsPerSample = audioFile.getBitDepth();

    aInfo.duration = audioFile.getLengthInSeconds();
    aInfo.totalSamples = audioFile.getNumSamplesPerChannel() * audioFile.getNumChannels();

    aInfo.CompressionMethod = "Uncompressed";

    return aInfo;
}
/*

helper function to get the audioOutput for the loading File Function
reading the .wav file using the AudioFile library
*/

audioReadOutput audioread(string fileName, ctlInfo ctl) {


    string fileNameR;

    fileNameR = fileName;

    ctlInfo ctlI;

    ctlI = ctl;

    AudioFile<float> audioFile;

    audioFile.load(fileNameR);
    
    //subtract one from the channel as there is no channel 0 but vectors start at 0
    int channel = ctlI.channel -1;
    int numSamples = audioFile.getNumSamplesPerChannel();
    int sampleRate = audioFile.getSampleRate();

    vector<float> audioSamples;

    for (int i = 0; i < numSamples; i++) {
        float currentSample = audioFile.samples[channel][i];
        audioSamples.push_back(currentSample);
    }

    audioReadOutput aro;
    aro.audiosamples = audioSamples;
    aro.SampleRate = sampleRate;

    return aro;
}

/*

loads the .wav file and does some necessary manipulations to recieve the signal in the correct form

such as up and down sampling to get to a sample rate of 16k
*/

loadWaveFileInputs loadWaweFile(ctlInfo ctl, FileInfo fInfo) {

    vector<float> audioSamples;

    loadWaveFileInputs lwaveFI;

    FileInfo fileInfo; 
    fileInfo = fInfo;

    ctlInfo ctlI;

    ctlI = ctl;
    
    const bool MAX = true;

    vector<float> maxAudioSamples;
    audioReadOutput aWawe;

    AudioInfo aInfo;

    vector<float> scaledAudioSamples;


    aInfo = getAudioInfo(fileInfo);

    if (aInfo.CompressionMethod != "Uncompressed") {
        cerr << "The .wav file needs to be uncompressed " << endl;
        throw std::runtime_error("File is compressed");
    }
    else if (aInfo.numChannels < ctlI.channel) {
        cerr << "The file does not contain the requested channel " << endl;
        throw std::runtime_error("Channel does not exist");
    }
    else if (aInfo.SampleRate != 8000 && aInfo.SampleRate != 16000 && aInfo.SampleRate != 24000 && aInfo.SampleRate != 32000 && aInfo.SampleRate != 48000) {
        cerr << "This .wav file does not have required sample rate 8, 16, 24, 32, or 48k" << endl;
        throw std::runtime_error("no valid sample rate");
    }
    else if ((aInfo.totalSamples / aInfo.SampleRate) < 3) {
        cerr << "This  .wav file has duration less than 3 seconds" << endl;
        throw std::runtime_error("file duration is too short");
    }
    else {
         aWawe = audioread(fileInfo.name,ctlI);
         audioSamples = aWawe.audiosamples;

        //Do added math to the audioSamples vector

        // Scale factor needed to align this code
        // with code that generated the CNN
        float scalingFactor = 32768 / 32767;
         scaledAudioSamples = multiVect(audioSamples, scalingFactor);

         maxAudioSamples = MaxOrMin(audioSamples, -1, MAX);

        //Do the correction to the audiosamples vector
        //get slice of channel number
        vector<float> absAudioSamples = vectorAbs(audioSamples);
        float sigCheck = 0.0;

        for (int i = 0; i < absAudioSamples.size(); ++i) {
            sigCheck += absAudioSamples.at(i);
        }

        fileInfo.duration = aInfo.duration;
        fileInfo.sampleRate = aInfo.SampleRate;
        ctlI.sampleRate = aInfo.SampleRate;


        if (sigCheck == 0) {

            cerr << "This .wav file has no signal" << endl;
            throw std::runtime_error("no signal found");
            
        }
        //sample rate conversions to 16k
        else {
            if (fileInfo.sampleRate == 8000) {
                maxAudioSamples = upSampleByTwo(maxAudioSamples); 
            }
            else if (fileInfo.sampleRate == 24000) {
                maxAudioSamples = upSampleByTwo(maxAudioSamples);
                maxAudioSamples = downSampleByThree(maxAudioSamples);
            }
            else if (fileInfo.sampleRate == 32000) {
                maxAudioSamples = downSampleByTwo(maxAudioSamples);
            }
            else if (fileInfo.sampleRate == 48000) {
                maxAudioSamples = downSampleByThree(maxAudioSamples);
            }
        }
        lwaveFI.audiosamples = maxAudioSamples;
        lwaveFI.fileInfo = fileInfo;
        lwaveFI.ctl = ctlI;
    }

    return lwaveFI;
    
}



/*

helper function for the level Meter which runs the signal through a IIR filter

*/



vector<float> IIRfilter(float bCoeff, vector<float> aCoeffs, vector<float> inS) {


    vector<float> inSignal = inS;
    vector<float> outsignal;
    vector<float> aCoeffsV;
    aCoeffsV = aCoeffs;

    float bCoeffF;
    bCoeffF = bCoeff;

    int nIn;
    int inSignalP;

    nIn = inSignal.size();

    auto it = inSignal.insert(inSignal.begin(), 0);

    inSignal.insert(it, 0);

    inSignalP = nIn + 2;
    outsignal = getZeroVector( inSignalP);

   
    vector<float> sliced;

    for (int i = 2; i < outsignal.size(); ++i) {

        //slices out a portion of the outsignal array between i-2 and i-1
        sliced = slice(outsignal, i - 2, i);
        outsignal.at(i) = inSignal.at(i) * bCoeffF - dotMulti(aCoeffsV, sliced);
    }
    return slice(outsignal, 2, outsignal.size());


}


/*

helper function for the AudioNormalization

*/


Speech levelMeter(vector<float> inSamplesA, ctlInfo ctl) {

    const float logConstant =  20 * log10(2);
    vector<float> logEnv;
    float hangSamples;
    float sumS;
    vector<float> filterSamples;
    vector<float> inSamplesAbs;
    vector<float> inSamples;
    vector<float> downSampledSamples;
    const bool MAX = true;
    const bool MIN = false;

    inSamples = inSamplesA;
    float value = -1 / (ctl.sampleRate * 0.03);
    float gFilter = exp(value);
    sumS = sumSquares(inSamples);
    inSamplesAbs = vectorAbs(inSamples);

    auto val = 1 - gFilter;
    auto val2 = -2 * gFilter;
    auto val3 = pow(gFilter, 2);
    float bCoeff = pow(val, 2);
    vector<float> aCoeff;
    aCoeff.push_back(val3);
    aCoeff.push_back(val2);

    filterSamples = IIRfilter(bCoeff, aCoeff, inSamplesAbs);

    float downSampleFactor = round(ctl.sampleRate / 500);

    downSampledSamples = multiSlice(filterSamples, downSampleFactor);

    int nSamples = downSampledSamples.size();

    hangSamples = round(0.2 * 500);

    sumS = sumS / downSampleFactor;


    float maxV = pow((float)2, -20);

    //getting the max
    logEnv = MaxOrMin(downSampledSamples,maxV, MAX);

    logEnv = log2Vect(logEnv);

    logEnv = floorVect(logEnv);

    //getting the min
    logEnv = MaxOrMin(logEnv, 1, MIN);

    vector<vector<float>> activityMatrix;

    for (int i = 0; i < nSamples; ++i) {
        vector<float> v = getZeroVector(16);
        activityMatrix.push_back(v);
    }


    for (int sample = 0; sample < nSamples; sample++) {
        auto lastSample = min((float)nSamples, sample + hangSamples); //-1
        
        for (int i = sample; i < lastSample; ++i) {
            for (int j = 0; j < logEnv.at(sample) + 16; j++) {
                (activityMatrix.at(i)).at(j) = 1;
            }
        }
       
    }
    vector<float> logActivity;
    vector<float> logDiff;
    float activeSpeechLevel;

    logActivity = getZeroVector(16);
    logDiff = getZeroVector(16);


    for (int i = 0; i < logActivity.size(); ++i) {
        float totalActive; 

        totalActive = sumCol(activityMatrix, i);

        if (0.0 < totalActive) {
            float div = sumS / totalActive;
            float logd = log10(div);
            logActivity.at(i) = 10 * logd;
            int j = i - 15; 
            logDiff.at(i) = logActivity.at(i) -(j) *logConstant;
        }
        else {
            logActivity.at(i) = 100;
            logDiff.at(i) = 100;
        }
    }


    activeSpeechLevel = -100;
    
    for (int i = 0; i < logDiff.size()-1; ++i) {
        int next = i + 1;
        if (logDiff.at(i) >= 15.9 && logDiff.at(next) <= 15.9) {
            if (logDiff.at(i) == logDiff.at(next)) {
                activeSpeechLevel = logActivity.at(i);
            }

            else {
                float logD = logDiff.at(i) - logDiff.at(next);
                float scaledStep = (logDiff.at(i) - 15.9) / logD;
                activeSpeechLevel = ((1 - scaledStep) * logActivity.at(i)) + (scaledStep * logActivity.at(next));
            }
        }
    }
    float speechActivityFactor;
    if (activeSpeechLevel > -100) {
        speechActivityFactor = (sumS / nSamples) * pow(10, (-activeSpeechLevel / 10));
    }
    else {
        speechActivityFactor = 0;
    }

    Speech speech;

    speech.speechActivityFactor = speechActivityFactor;
    speech.activeSpeechLevel = activeSpeechLevel;      

    return speech;


}


/*
AudioNormalize function
*/
audioNormalizeOutput audioNormalize(vector<float> cAudio, ctlInfo ctl, FileInfo fInfo) {

    vector<float> cAudioV;
    cAudioV = cAudio;
    vector<float> cAudioNoMean;
    FileInfo fileInfo;
    fileInfo = fInfo;
    ctlInfo ctlI;
    ctlI = ctl;
    Speech outSpeech;

    int targetLevel = -26;
    int nSamples = cAudioV.size();
    
    float sum = 0.0;

    sum = sumVector(cAudioV);

    float mean = sum / nSamples;
    cAudioNoMean = removeMean(cAudioV, mean); 
   
    outSpeech = levelMeter(cAudioNoMean, ctlI);
    fileInfo.activeLevel = outSpeech.activeSpeechLevel;
    fileInfo.speechActivityFactor = outSpeech.speechActivityFactor;

    audioNormalizeOutput aOutput;
    aOutput.fileinfo = fileInfo;
    
    if (ctlI.levelNormalization == 1) {
        float sub = ((float)targetLevel - fileInfo.activeLevel)/20;
        float gain = pow(10,sub);
        aOutput.outSamples = multiVect(cAudioNoMean, gain);
    }
    else {
        aOutput.outSamples = cAudioNoMean;
    }
    return aOutput;

}


/*
Main Function for WaweNet
*/
void WAWEnet(vector<string> fileA) {

    FILE* out;

    string WaveFileResults = "";
    vector<string> fileAndArgs;
    fileAndArgs = fileA;
    ctlInfo ctl;
    vector<string> inFileList;
    vector<float> audioSamples;
    int nAudioFiles;
    inFileList.push_back(" ");



    string fileName = fileAndArgs.at(0);



    inFileList = processInFile(fileName);
    nAudioFiles = inFileList.size();


    ctl = parseVars(slice(fileAndArgs, 1, fileAndArgs.size()), ctl);
    ctl.segLength = 48000;
    ctl.activityThreshold = 0.45;



    ctl.outFileHeader = makeHeader();



    //Open output file if one exists
    if (!ctl.outFileName.empty()) {
        try {
            out = fopen((ctl.outFileName).c_str(), "a");
        }
        catch (int e) {
            cerr << "could not open output file, does it exist?" << endl;
            throw std::runtime_error("file not found");
        }
        fprintf(out, ctl.outFileHeader.c_str());
    }

    FileInfo fileInfo;


    for (int cAudioFile = 0; cAudioFile < nAudioFiles; ++cAudioFile) {

        fileInfo.name = inFileList.at(cAudioFile);
        loadWaveFileInputs loadFile = loadWaweFile(ctl, fileInfo);
        if (cAudioFile == 0) {
            cout << ctl.outFileHeader << endl;
        }

         fileInfo = loadFile.fileInfo;
         audioSamples = loadFile.audiosamples;
         ctl = loadFile.ctl;

        if (fileInfo.exception.empty()) {
            int nAudioSamples = audioSamples.size();


            int nSegments = floor((nAudioSamples - ctl.segLength) / ctl.segStep) + 1;      

            vector<float> netOut;
            vector<float> exceedsActivityThreshold;
            vector<float> currentAudioSamples;
            audioNormalizeOutput normOutput;
            vector<float> normalizedInput;

            int firstSample = 0;

            for (int CurSeg = 0; CurSeg < nSegments; ++CurSeg) {
                int lastSample = firstSample + ctl.segLength; 
                currentAudioSamples = slice(audioSamples, firstSample, lastSample);
                normOutput = audioNormalize(currentAudioSamples, ctl, fileInfo);
                
                normalizedInput = normOutput.outSamples;
                float net = getWAWEnetCNN(normalizedInput, ctl.WAWEnetMode);
               
                // store results
                netOut.push_back(net);
                fileInfo.allActivityFactors.push_back(normOutput.fileinfo.speechActivityFactor);
                fileInfo.allActiveLevels.push_back(normOutput.fileinfo.activeLevel);
                if (normOutput.fileinfo.speechActivityFactor > ctl.activityThreshold) {
                    exceedsActivityThreshold.push_back(net);
                }

                // update pointer to first audio sample of the next segment
                firstSample = firstSample + ctl.segStep;
            }
            // hmmm, this only stores the SAF and active level for the last speech segment :(
            fileInfo.speechActivityFactor = normOutput.fileinfo.speechActivityFactor;
            fileInfo.activeLevel = normOutput.fileinfo.activeLevel;
            fileInfo.netOut = netOut;

            // average the outputs over all segments where the speech activity factor
            // meets or exceeds threshold and store it
            fileInfo.grandMean = meanVector(exceedsActivityThreshold);

            // generate the text to print to the screen
            WaveFileResults = createOutputString(fileInfo, ctl);

        }
        else {
            cerr << "fileInfo exception" << endl;
            throw std::runtime_error("fileInfo exception");
            
        }
        WaveFileResults += "\n";
        WaveFileResults += "\n";
        if (!ctl.outFileName.empty()) {
            fprintf(out, WaveFileResults.c_str());
            if (cAudioFile + 1 >= nAudioFiles) {
                fclose(out);
            }
        }
        std::cout << WaveFileResults << std::endl;

    }
}
