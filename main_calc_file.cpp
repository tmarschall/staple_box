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

using namespace std;

const double D_PI = 3.14159265358979;

int main()
{
  cout << "Data file: ";
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
  cout << "Start gamma: " ;
  double dStart;
  cin >> dStart;
  cout << dStart << "\nEnding gamma: ";
  double  dStop;
  cin >> dStop;
  cout << dStop << "\nStrain Rate: ";
  double dStrainRate;
  cin >> dStrainRate;
  cout << dStrainRate << "\nIntegration step size: ";
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
  cout << dDR << endl;

  if (dStressSaveRate < dStrainRate * dStep)
    dStressSaveRate = dStrainRate * dStep;
  if (dPosSaveRate < dStrainRate)
    dPosSaveRate = dStrainRate;

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

  cout << "Loading file: " << strFile << endl;
  DatFileInput cData(szFile, 0);
  int nRows = cData.getRows();
  if (nRows != nStaples) {
    cerr << "Number of rows does not match number of staples" << endl;
    exit(1);
  }
  
  dL = cData.getFloat(0,7);
  dPacking = cData.getFloat(0,8);
  dGamma = cData.getFloat(0,9);
  dTotalGamma = cData.getFloat(0,10);

  
  cData.getColumn(dRad, 1);
  cData.getColumn(dAs, 2);
  cData.getColumn(dAb, 3);
  cData.getColumn(dX, 4);
  cData.getColumn(dY, 5);
  cData.getColumn(dPhi, 6);
  cout << "Data loaded " << endl;
  
  int tStart = time(0);
  
  Staple_Box *cStaples;
  cStaples = new Staple_Box(nStaples, dL, dX, dY, dPhi, dRad, dAs, dAb, dDR);
  
  cout << "Staples initialized" << endl;
  (*cStaples).set_gamma(dGamma);
  (*cStaples).set_total_gamma(dTotalGamma);

  (*cStaples).set_strain(dStrainRate);
  (*cStaples).set_step(dStep);
  (*cStaples).set_data_dir(strDir);
  cout << "Configuration set" << endl;
  (*cStaples).split_staples();
  (*cStaples).find_neighbors();
  (*cStaples).calculate_stress_energy();
  (*cStaples).display(1,1,1,1);
  //(*cStaples).run_strain(dStart, dStop, dStressSaveRate, dPosSaveRate);
  //(*cStaples).display(1,0,0,1);

  int tStop = time(0);
  cout << "\nRun Time: " << tStop - tStart << endl;

  delete[] dX; delete[] dY; delete[] dPhi; 
  delete[] dRad; delete[] dAs; delete[] dAb;
  delete cStaples;

  return 0;
}
