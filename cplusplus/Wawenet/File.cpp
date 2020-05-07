// Wavenet.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "WAWEnet.h"



void printhelp() {
    string wavFiles;
    ifstream infile;
    try {
        infile.open("helpScreen.txt");
    }
    catch (std::ios_base::failure& e) {
        cerr << "could not find helpfile" << '\n';
    }

    while (getline(infile, wavFiles))
    {
        cout << wavFiles << endl;
    }
    infile.close();
}



int main(int argc , char* argv[])

{

  vector<string> fileArgs;
    
  if (argc > 1) {
    fileArgs.assign(argv + 1, argv + argc);
  }
  else {
      printhelp();
      return 0;
  }


  if (fileArgs.at(0) == "-h") {
      printhelp();
      return 0;
  }

  WAWEnet(fileArgs);
  
  return 0;

}


