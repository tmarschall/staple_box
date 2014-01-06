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
    dPacking = float_input(argc, argv, ++argn, "Packing Fraction");
    const double pi = 3.141592653589793;
    double dArea = nStaples*(12. + 3.*pi*0.25);
    if (dPacking > 0.25)
      dL = sqrt(dArea / 0.25);
    else
      dL = sqrt(dArea / dPacking);
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
    unsigned int path_n = strFile.find_last_of("/");
    if (strFile.substr(0,path_n+1) != strDir) {
      cerr << "File " << strFile << " does not match directory " << strDir << endl;
      exit(1);
    }
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
    if (dPacking > 0.455) {
      (*cStaples).simple_shrink_box(10000, 5e-5, 0.025, 0.45, 10, 50);
      (*cStaples).set_strain(1e-4);
      (*cStaples).run_strain(0, 10, dStressSaveRate, dPosSaveRate);
      (*cStaples).set_gamma(0);
      (*cStaples).set_total_gamma(0);
      (*cStaples).set_strain(0);
      if (dPacking > 0.505) {
	(*cStaples).simple_shrink_box(10000, 5e-5, 0.025, 0.5, 10, 50);
	(*cStaples).set_strain(1e-4);
	(*cStaples).run_strain(0, 10, dStressSaveRate, dPosSaveRate);
	(*cStaples).set_gamma(0);
	(*cStaples).set_total_gamma(0);
	(*cStaples).set_strain(0);
	if (dPacking > 0.545) {
	  (*cStaples).simple_shrink_box(10000, 2.5e-5, 0.025, 0.54, 10, 50);
	  (*cStaples).set_strain(1e-4);
	  (*cStaples).run_strain(0, 10, dStressSaveRate, dPosSaveRate);
	  (*cStaples).set_gamma(0);
	  (*cStaples).set_total_gamma(0);
	  (*cStaples).set_strain(0);
	  if (dPacking > 0.575) {
	    (*cStaples).simple_shrink_box(10000, 2.5e-5, 0.025, 0.57, 10, 50);
	    (*cStaples).set_strain(1e-4);
	    (*cStaples).run_strain(0, 10, dStressSaveRate, dPosSaveRate);
	    (*cStaples).set_gamma(0);
	    (*cStaples).set_total_gamma(0);
	    (*cStaples).set_strain(0);
	  }
	}
      }
    }
    (*cStaples).simple_shrink_box(12000, 2.5e-5, 0.025, dPacking, 10, 50);
    cout << "Box compressed to packing: " << dPacking << endl;
    (*cStaples).calculate_stress_energy();
    (*cStaples).display(1,0,0,1);
  }

  (*cStaples).set_strain(dStrainRate);
  (*cStaples).set_step(dStep);
  (*cStaples).set_data_dir(strDir);
  cout << "Configuration set" << endl;
  (*cStaples).split_staples();
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
