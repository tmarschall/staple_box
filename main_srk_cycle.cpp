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
#include <sstream>

using namespace std;

const double D_PI = 3.14159265358979;

int main()
{
  cout << "Data file ('r' for random): ";
  string strFile;
  getline(cin, strFile);
  const char* szFile = strFile.c_str();
  cout << strFile << "\nOutput Data Directory (including closing '/'): ";
  string strDir;
  getline(cin, strDir);
  const char* szDir = strDir.c_str();
  cout << strDir << "\nNumber of particles: ";
  int nStaples;
  cin >> nStaples;
  cout << nStaples << endl;

  //user options
  cout << "Min Packing: " ;
  double dMin;
  cin >> dMin;
  cout << dMin << "\nMax packing: ";
  double  dMax;
  cin >> dMax;
  cout << dMax << "\nShrink Rate: ";
  double dShrinkRate;
  cin >> dShrinkRate;
  cout << dShrinkRate << "\nIntegration step size: ";
  double dStep;
  cin >> dStep;
  cout << dStep << "\nPosition data save rate: ";
  double dPosSaveRate;
  cin >> dPosSaveRate;
  cout << dPosSaveRate << "\nStress data save rate: ";
  double dStressSaveRate;
  cin >> dStressSaveRate;
  cout << dStressSaveRate << "\nDR: ";
  double dDR;
  cin >> dDR;
  cout << dDR << "\nCycles: ";
  int nCycles;
  cin >> nCycles;
  cout << nCycles << endl;

  if (dStressSaveRate < dShrinkRate * dStep)
    dStressSaveRate = dShrinkRate * dStep;
  if (dPosSaveRate < dShrinkRate)
    dPosSaveRate = dShrinkRate;

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
  bool bShrink;
  unsigned long int nTime;
  if (strFile == "r")
  {
    bShrink = 1;
    const double pi = 3.141592653589793;
    double dArea = nStaples*(12. + 3.*pi*0.25);
    dL = sqrt(dArea / 0.2);
    cout << "Box length L: " << dL << endl;
    dRMax = 0.5;
    dAMax = 2.0;
    /*
    srand(time(0) + static_cast<int>(1000*dPacking));
    for (int p = 0; p < nStaples; p++)
    {
      dX[p] = dL * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dY[p] = dL * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dPhi[p] = 2.*pi * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dRad[p] = 0.5;
      dAs[p] = 2.;
      dAb[p] = 2.;   
    }
    */
    dGamma = 0.;
    dTotalGamma = 0.;
    nTime = 0;
  }
  else
  {
    cout << "Shrink or expand (1 or 0): ";
    cin >> bShrink;
    cout << bShrink << endl;
    cout << "Loading file: " << strFile << endl;
    DatFileInput cData(szFile, 0);
    int nRows = cData.getRows();
    if (nRows != nStaples) {
      cerr << "Number of rows does not match number of staples" << endl;
      exit(1);
    }

    dL = cData.getFloat(0,7);
    //dPacking = cData.getFloat(0,8);
    dGamma = cData.getFloat(0,9);
    dTotalGamma = cData.getFloat(0,10);

    cData.getColumn(dRad, 1);
    cData.getColumn(dAs, 2);
    cData.getColumn(dAb, 3);
    cData.getColumn(dX, 4);
    cData.getColumn(dY, 5);
    cData.getColumn(dPhi, 6);
    cout << "Data loaded " << endl;

    int nLen = strFile.length();
    string strTime = strFile.substr(nLen - 14, 10);
    stringstream ss(strTime);
    ss >> nTime;
  }

  if (bShrink)
    dPacking = dMax;
  else
    dPacking = dMin;
  
  int tStart = time(0);

  Staple_Box *cStaples;
  if (strFile == "r")
    cStaples = new Staple_Box(nStaples, dL, dRMax, dAMax, dDR);
  else
    cStaples = new Staple_Box(nStaples, dL, dX, dY, dPhi, dRad, dAs, dAb, dDR);

  cout << "Staples initialized" << endl;
  (*cStaples).set_gamma(dGamma);
  (*cStaples).set_total_gamma(dTotalGamma);
  (*cStaples).set_strain(0);
  (*cStaples).set_step(dStep);
  (*cStaples).set_data_dir(strDir);
  cout << "Configuration set" << endl;

  if (strFile == "r") {
    (*cStaples).reorder_particles();
    (*cStaples).reset_IDs();
    cout << "Particles reordered and IDs reset" << endl;
    (*cStaples).display(0,0,1,0);
    (*cStaples).find_neighbors_2();
    cout << "Neighbor check: " << endl;
    (*cStaples).display(0,0,1,0);
    (*cStaples).calculate_stress_energy();
    cout << "Stresses calculated" << endl;
    (*cStaples).display(1,0,1,1);
  }
  
 
  (*cStaples).set_strain(0);
  (*cStaples).set_data_dir(strDir);
  unsigned int nPSave = int(0.001 / dShrinkRate + 0.5);
  if (bShrink) {
    nTime = (*cStaples).simplest_shrink_box(nTime, dShrinkRate, dStep, dMax, 500, nPSave, 1);
  }
  for (int c = 0; c < nCycles; c++) {
    nTime = (*cStaples).simplest_expand_box(nTime, dShrinkRate, dStep, dMin, 500, nPSave, 1);
    nTime = (*cStaples).simplest_shrink_box(nTime, dShrinkRate, dStep, dMax, 500, nPSave, 1);
  }

  int tStop = time(0);
  cout << "\nRun Time: " << tStop - tStart << endl;

  delete[] dX; delete[] dY; delete[] dPhi; 
  delete[] dRad; delete[] dAs; delete[] dAb;
  delete cStaples;

  return 0;
}
