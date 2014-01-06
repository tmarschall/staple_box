#include "staple_box.h"
#include "file_input.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sstream>

using namespace std;

const double D_PI = 3.14159265358979;

int int_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    int input = atoi(argv[argn]);
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    int input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}
double float_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    double input = atof(argv[argn]);
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    double input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}
string string_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    string input = argv[argn];
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    string input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}

int main(int argc, char *argv[])
{
  int argn = 0;
  string strInputDir = string_input(argc, argv, ++argn, "Input directory");
  string strOutputDir = string_input(argc, argv, ++argn, "Output directory");
  int nSpace = int_input(argc, argv, ++argn, "File spacing");
  int nFiles = int_input(argc, argv, ++argn, "Number of files");
  
  for (int f = 0; f <= nFiles*nSpace; f += nSpace) {
    stringstream ssF;
    ssF << setw(10) << setfill('0') << f;
    string strInf = strInputDir + "/sd" + ssF.str() + ".dat";
    DatFileInput cData(strInf.c_str());
    int nParticles = cData.getRows();
    double dL = cData.getFloat(0,7);
    double dPack = cData.getFloat(0,8);
    double dGamma = cData.getFloat(0,9);
    double dTotalGamma = cData.getFloat(0,10);
    double *pdX = new double[nParticles];
    double *pdY = new double[nParticles];
    double *pdPhi = new double[nParticles];
    double *pdR = new double[nParticles];
    double *pdAs = new double[nParticles];
    double *pdAb = new double[nParticles];

    cData.getColumn(pdX, 4);
    cData.getColumn(pdY, 5);
    cData.getColumn(pdPhi, 6);
    cData.getColumn(pdR, 1);
    cData.getColumn(pdAs, 2);
    cData.getColumn(pdAb, 3);
    
    Staple_Box cStaples(nParticles, dL, pdX, pdY, pdPhi, pdR, pdAs, pdAb, 0.1, 12, 24);
    cStaples.set_gamma(dGamma);
    cStaples.set_total_gamma(dTotalGamma);
    cStaples.set_data_dir(strOutputDir);
    cStaples.set_strain(0);
    cStaples.set_step(0.01);
    
    cStaples.split_staples();
    cStaples.find_neighbors();
    cStaples.relax_box(f, 1000000, 0.005, 20, 100);
    
  }

  return 0;
}
