/*
 *
 *
 */

#include "staple_box.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include "file_input.h"
#include <string>
#include <fstream>

using namespace std;

const double D_PI = 3.14159265358979;


// input functions that parse command line argument, then prompt for input if end of cl arguments has been reached
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


int main(int argc, char* argv[])
{
  int argn = 0;
  string strFile = string_input(argc, argv, ++argn, "Data file ('r' for random)");
  const char* szFile = strFile.c_str();
  string strDir = string_input(argc, argv, ++argn, "Output Data Directory (including closing '/')");
  const char* szDir = strDir.c_str();
  int nStaples = int_input(argc, argv, ++argn, "Number of particles");

  //user options
  double dStart = float_input(argc, argv, ++argn, "Start gamma");
  double dStop = float_input(argc, argv, ++argn, "Ending gamma");
  double dStrainRate = float_input(argc, argv, ++argn, "Strain rate");
  double dStep = float_input(argc, argv, ++argn, "Integration step size");
  double dPosSaveRate = float_input(argc, argv, ++argn, "Position save rate (gamma)");
  double dStressSaveRate = float_input(argc, argv, ++argn, "Stress save rate (gamma)");
  double dDR = float_input(argc, argv, ++argn, "Cell, neighbor list padding DR");
  bool bSaveF = (bool)int_input(argc, argv, ++argn, "Save forces during run");

  if (dStressSaveRate < dStrainRate * dStep)
    dStressSaveRate = dStrainRate * dStep;
  if (dPosSaveRate < dStrainRate)
    dPosSaveRate = dStrainRate;
  int nPosSaveRate = int(dPosSaveRate / dStrainRate + 0.5);
  int nStart = int(dStart / dStrainRate + 0.5);
  int nStop = int(dStop / dStrainRate + 0.5);

  double dL;
  double *dX = new double[nStaples];
  double *dY = new double[nStaples];
  double *dPhi = new double[nStaples];
  double *dRad = new double[nStaples];
  double *dAs = new double[nStaples];
  double *dAb = new double[nStaples];
  double dRMax;
  double dAMax;
  double dGamma;
  double dTotalGamma;
  double dPacking;
  DatFileInput *cData = 0;
  Staple_Box *cStaples = 0;

  char szNum[11];
  unsigned int path_n = strFile.find_last_of("/");
  if (strFile.substr(0,path_n+1) != strDir) {
    cerr << "File " << strFile << " does not match directory " << strDir << endl;
    exit(1);
  }
  if (atoi(strFile.substr(strFile.length()-14,strFile.length()-4).c_str()) != nStart) {
    cerr << "Start file " << strFile << " does not match start n " << nStart << endl;
  }

  string strOutFile = strDir;
  strOutFile.append("/av.dat");
  ofstream outf(strOutFile.c_str());
  
  for (int n = nStart; n <= nStop; n+= nPosSaveRate) {
    sprintf(szNum, "%010d", n);
    strFile.replace(strFile.length()-14, 10, szNum);
    cout << "Loading file: " << strFile << endl;
    delete cData;
    cData = new DatFileInput(szFile, 0);
    int nRows = cData->getRows();
    if (nRows != nStaples) {
      cerr << "Number of rows does not match number of staples" << endl;
      exit(1);
    }
  
    dL = cData->getFloat(0,7);
    dPacking = cData->getFloat(0,8);
    dGamma = cData->getFloat(0,9);
    dTotalGamma = cData->getFloat(0,10);
    /*
    if (dTotalGamma < dStart - 0.1*dPosSaveRate || dTotalGamma > dStart + 0.1*dPosSaveRate) {
      cerr << "Total gamma of file does not match input" << endl;
      exit(1);
    }
    */
  
    cData->getColumn(dRad, 1);
    cData->getColumn(dAs, 2);
    cData->getColumn(dAb, 3);
    cData->getColumn(dX, 4);
    cData->getColumn(dY, 5);
    cData->getColumn(dPhi, 6);
    cout << "Data loaded " << endl;
    
    //int tStart = time(0);

    delete cStaples;
    cStaples = new Staple_Box(nStaples, dL, dX, dY, dPhi, dRad, dAs, dAb, dDR);

    cout << "Staples initialized" << endl;
    cStaples->set_gamma(dGamma);
    cStaples->set_total_gamma(dTotalGamma);


    cStaples->set_strain(dStrainRate);
    cStaples->set_step(dStep);
    cStaples->set_data_dir(strDir);
    cout << "Configuration set" << endl;
    cStaples->split_staples();
    cStaples->find_neighbors();
    cStaples->calculate_stress_energy();
    cStaples->display(0,0,0,1);
    
    //double *pdFt = cStaples.getFt();
    //double *pdMOI = cStaples.getMOI();
    //double dWel = 0.0;
    //for (int p = 0; p < nStaples; p++) {
    //  dWel += pdFt[p] / pdMOI[p];
    //}
    //dWel /= nStaples;
    
    cStaples->save_staple_forces(n);
    //outf << dTotalGamma << " " << dWel << endl;

    //(*cStaples).display(1,0,0,1);
    //(*cStaples).run_strain(dStart, dStop, dStressSaveRate, dPosSaveRate);
    //(*cStaples).display(1,0,0,1);
    
    //int tStop = time(0);
    //cout << "\nRun Time: " << tStop - tStart << endl;

    //delete cStaples;
  }

  outf.close();

  delete[] dX; delete[] dY; delete[] dPhi; 
  delete[] dRad; delete[] dAs; delete[] dAb;
  delete cStaples; delete cData;

  return 0;
}
