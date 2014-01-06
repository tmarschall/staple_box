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
  cout << "Start gamma dot: " ;
  double dStart;
  cin >> dStart;
  cout << dStart << "\nMinimum gamma dot: ";
  double  dStop;
  cin >> dStop;
  cout << dStop << "\nRate of change: ";
  double dChangeRate;
  cin >> dChangeRate;
  cout << dChangeRate << "\nIntegration step size: ";
  double dStep;
  cin >> dStep;
  cout << dStep << "\nPosition data save interval: ";
  int nPosSaveRate;
  cin >> nPosSaveRate;
  cout << nPosSaveRate << "\nStress data save interval: ";
  int nStressSaveRate;
  cin >> nStressSaveRate;
  cout << nStressSaveRate << "\nDR: ";
  double dDR;
  cin >> dDR;
  cout << dDR << endl;


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
  if (strFile == "r")
  {
    cout << "Packing Fraction: ";
    cin >> dPacking;
    cout << dPacking << endl;
    const double pi = 3.141592653589793;
    double dArea = nStaples*(12. + 3.*pi*0.25);
    dL = sqrt(dArea / 0.25);
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
  }
  else
  {
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
    if (dTotalGamma < dStart - 0.5*dPosSaveRate || dTotalGamma > dStart + 0.5*dPosSaveRate) {
      cerr << "Total gamma of file does not match input" << endl;
      exit(1);
    }

    cData.getColumn(dRad, 1);
    cData.getColumn(dAs, 2);
    cData.getColumn(dAb, 3);
    cData.getColumn(dX, 4);
    cData.getColumn(dY, 5);
    cData.getColumn(dPhi, 6);
    cout << "Data loaded " << endl;
  }
  
  int tStart = time(0);

  Staple_Box *cStaples;
  if (strFile == "r")
    cStaples = new Staple_Box(nStaples, dL, dRMax, dAMax, dDR);
  else
    cStaples = new Staple_Box(nStaples, dL, dX, dY, dPhi, dRad, dAs, dAb, dDR);

  cout << "Staples initialized" << endl;
  (*cStaples).set_gamma(dGamma);
  (*cStaples).set_total_gamma(dTotalGamma);

  if (strFile == "r") {
    (*cStaples).reorder_particles();
    (*cStaples).reset_IDs();
    cout << "Particles reordered and IDs reset" << endl;
    (*cStaples).calculate_stress_energy();
    cout << "Stresses calculated" << endl;
    (*cStaples).display(1,0,1,1);
    
    cout << "Compressing box" << endl;
    string strPDir = strDir + "srk/";
    cout << "Writing shrinking data to: " << strPDir << endl;
    (*cStaples).set_data_dir(strPDir);
    (*cStaples).set_strain(0);
    (*cStaples).simple_shrink_box(12000, 2.5e-5, 0.025, dPacking, 10, 50);
    cout << "Box compressed to packing: " << dPacking << endl;
    (*cStaples).calculate_stress_energy();
    (*cStaples).display(1,0,0,1);
  }

  (*cStaples).set_strain(dStrainRate);
  (*cStaples).set_step(dStep);
  (*cStaples).set_data_dir(strDir);
  cout << "Configuration set" << endl;
  (*cStaples).find_neighbors();
  (*cStaples).calculate_stress_energy();
  (*cStaples).display(1,0,0,1);
  (*cStaples).run_strain(dStart, dStop, dStressSaveRate, dPosSaveRate);
  (*cStaples).display(1,0,0,1);

  int tStop = time(0);
  cout << "\nRun Time: " << tStop - tStart << endl;

  delete[] dX; delete[] dY; delete[] dPhi; 
  delete[] dRad; delete[] dAs; delete[] dAb;
  delete cStaples;

  return 0;
}
