/*  staple_box.cpp
 *
 *
 */

#include "staple_box.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <algorithm>
#include "cudaErr.h"
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

const double D_PI = 3.14159265358979;


void Staple_Box::reconfigure_cells()
{
  double dWMin = 2.24 * (m_dRMax + m_dAMax) + m_dEpsilon;
  double dHMin = 2 * (m_dRMax + m_dAMax) + m_dEpsilon;

  int nNewCellRows = max(static_cast<int>(m_dL / dHMin), 1);
  int nNewCellCols = max(static_cast<int>(m_dL / dWMin), 1);
  if (nNewCellRows != m_nCellRows || nNewCellCols != m_nCellCols) {
    delete[] h_pnPPC; delete[] h_pnCellList; delete[] h_pnAdjCells;
    cudaFree(d_pnPPC); cudaFree(d_pnCellList); cudaFree(d_pnAdjCells);
#if GOLD_FUNCS == 1
    delete[] g_pnPPC; delete[] g_pnCellList; delete[] g_pnAdjCells;
#endif
    *h_bNewNbrs = 1;
    configure_cells();
  }
  else {
    m_dCellW = m_dL / m_nCellCols;
    m_dCellH = m_dL / m_nCellRows;
  }
}

// Just setting things up here
//  configure_cells() decides how the space should be divided into cells
//  and which cells are next to each other
void Staple_Box::configure_cells()
{
  assert(m_dRMax > 0.0);
  assert(m_dAMax >= 0.0);

  // Minimum height & width of cells
  //  Width is set so that it is only possible for particles in 
  //  adjacent cells to interact as long as |gamma| < 0.5
  double dWMin = 2.24 * (m_dRMax + m_dAMax) + m_dEpsilon;
  double dHMin = 2 * (m_dRMax + m_dAMax) + m_dEpsilon;

  m_nCellRows = max(static_cast<int>(m_dL / dHMin), 1);
  m_nCellCols = max(static_cast<int>(m_dL / dWMin), 1);
  m_nCells = m_nCellRows * m_nCellCols;
  cout << "Cells: " << m_nCells << ": " << m_nCellRows << " x " << m_nCellCols << endl;

  m_dCellW = m_dL / m_nCellCols;
  m_dCellH = m_dL / m_nCellRows;
  cout << "Cell dimensions: " << m_dCellW << " x " << m_dCellH << endl;

  h_pnPPC = new int[m_nCells];
  h_pnCellList = new int[m_nCells * m_nMaxPPC];
  h_pnAdjCells = new int[8 * m_nCells];
  cudaMalloc((void **) &d_pnPPC, m_nCells * sizeof(int));
  cudaMalloc((void **) &d_pnCellList, m_nCells*m_nMaxPPC*sizeof(int));
  cudaMalloc((void **) &d_pnAdjCells, 8*m_nCells*sizeof(int));
  m_nDeviceMem += m_nCells*(9+m_nMaxPPC)*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnPPC = new int[m_nCells];
  g_pnCellList = new int[m_nCells * m_nMaxPPC];
  g_pnAdjCells = new int[8 * m_nCells];
#endif
  
  // Make a list of which cells are next to each cell
  // This is done once for convinience 
  for (int c = 0; c < m_nCells; c++)
    {
      int nRow = c / m_nCellCols; 
      int nCol = c % m_nCellCols;

      int nAdjCol1 = (nCol + 1) % m_nCellCols;
      int nAdjCol2 = (m_nCellCols + nCol - 1) % m_nCellCols;
      h_pnAdjCells[8 * c] = nRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 1] = nRow * m_nCellCols + nAdjCol2;

      int nAdjRow = (nRow + 1) % m_nCellRows;
      h_pnAdjCells[8 * c + 2] = nAdjRow * m_nCellCols + nCol;
      h_pnAdjCells[8 * c + 3] = nAdjRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 4] = nAdjRow * m_nCellCols + nAdjCol2;
      
      nAdjRow = (m_nCellRows + nRow - 1) % m_nCellRows;
      h_pnAdjCells[8 * c + 5] = nAdjRow * m_nCellCols + nCol;
      h_pnAdjCells[8 * c + 6] = nAdjRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 7] = nAdjRow * m_nCellCols + nAdjCol2;
      
      //cout << "Cell " << c << ": " << h_pnAdjCells[8*c] << ", " << h_pnAdjCells[8*c+1] << ", " 
      //   << h_pnAdjCells[8*c+2] << ", " << h_pnAdjCells[8*c+3] << "," << h_pnAdjCells[8*c+4] << ", " 
      //   << h_pnAdjCells[8*c+5] << ", " << h_pnAdjCells[8*c+6] << ", " << h_pnAdjCells[8*c+7] << endl;
    }
  cudaMemcpy(d_pnAdjCells, h_pnAdjCells, 8*m_nCells*sizeof(int), cudaMemcpyHostToDevice);
#if GOLD_FUNCS == 1
  for (int c = 0; c < 8*m_nCells; c++) {
    g_pnAdjCells[c] = h_pnAdjCells[c];
  }
#endif
  checkCudaError("Configuring cells");
}

// Set the thread configuration for kernel launches
void Staple_Box::set_kernel_configs()
{
  switch (m_nStaples)
    {
    case 512:
      m_nGridSize = 4;
      m_nBlockSize = 128;
      m_nSpGridSize = 8;
      m_nSpBlockSize = 192;
      m_nSM_CalcF = 4*192*sizeof(double);
      m_nSM_CalcFSE = m_nSM_CalcF + 4*264*sizeof(double);
    case 1024:
      m_nGridSize = 8;  // Grid size (# of thread blocks)
      m_nBlockSize = 128; // Block size (# of threads per block)
      m_nSpGridSize = 16;
      m_nSpBlockSize = 192;
      m_nSM_CalcF = 4*192*sizeof(double);
      m_nSM_CalcFSE = m_nSM_CalcF + 4*264*sizeof(double); // Size of shared memory per block
      break;
    case 2048:
      m_nGridSize = 16;  // Grid size (# of thread blocks)
      m_nBlockSize = 128; // Block size (# of threads per block)
      m_nSpGridSize = 32;
      m_nSpBlockSize = 192;
      m_nSM_CalcF = 4*192*sizeof(double);
      m_nSM_CalcFSE = m_nSM_CalcF + 4*264*sizeof(double); // Size of shared memory per block
      break;
    case 4096:
      m_nGridSize = 16;
      m_nBlockSize = 256;
      m_nSpGridSize = 32;
      m_nSpBlockSize = 384;
      m_nSM_CalcF = 4*384*sizeof(double);
      m_nSM_CalcFSE = m_nSM_CalcF + 4*520*sizeof(double);
      break;
    default:
      m_nGridSize = 32;
      m_nBlockSize = 256;
      m_nSpGridSize = 32;
      m_nSpBlockSize = 384;
      m_nSM_CalcF = 4*384*sizeof(double);
      m_nSM_CalcFSE = m_nSM_CalcF + 4*520*sizeof(double);
    };
  cout << "Kernel config (staples):\n";
  cout << m_nGridSize << " x " << m_nBlockSize << endl;
  cout << "Kernel config (spherocylinders):\n";
  cout << m_nSpGridSize << " x " << m_nSpBlockSize << endl;
  cout << "Shared memory allocation (calculating forces):\n";
  cout << (float)m_nSM_CalcF / 1024. << "KB" << endl;
  cout << "Shared memory allocation (calculating S-E):\n";
  cout << (float)m_nSM_CalcFSE / 1024. << " KB" << endl; 
}

void Staple_Box::construct_defaults()
{
  m_dGamma = 0.0;
  m_dTotalGamma = 0.0;
  m_dStep = 1;
  m_dStrainRate = 1e-3;
  m_strDataDir = "./";
  m_strFileSE = "sd_stress_energy.dat";

  h_pdMOI = new double[m_nStaples];
  h_pdDtCoeffSin = new double[m_nStaples];
  h_pdDtCoeffCos = new double[m_nStaples];
  cudaHostAlloc((void**) &h_bNewNbrs, sizeof(int), 0);
  *h_bNewNbrs = 1;
  cudaMalloc((void**) &d_bNewNbrs, sizeof(int));
  cudaMemcpyAsync(d_bNewNbrs, h_bNewNbrs, sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_pdTempX, sizeof(double)*3*m_nStaples);
  cudaMalloc((void**) &d_pdTempY, sizeof(double)*3*m_nStaples);
  cudaMalloc((void**) &d_pdTempPhi, sizeof(double)*3*m_nStaples);
  cudaMalloc((void**) &d_pdXMoved, sizeof(double)*3*m_nStaples);
  cudaMalloc((void**) &d_pdYMoved, sizeof(double)*3*m_nStaples);
  cudaMalloc((void**) &d_pdCOM, sizeof(double)*m_nStaples);
  cudaMalloc((void**) &d_pdMOI, sizeof(double)*m_nStaples);
  cudaMalloc((void**) &d_pdDtCoeffSin, sizeof(double)*m_nStaples);
  cudaMalloc((void**) &d_pdDtCoeffCos, sizeof(double)*m_nStaples);
  m_bCOM = 0;
  m_nDeviceMem += 19*m_nStaples*sizeof(double);
#if GOLD_FUNCS == 1
  cout << "Allocating default arrays" << endl;
  g_bNewNbrs = new int;
  *g_bNewNbrs = 1;
  g_pdTempX = new double[3*m_nStaples];
  g_pdTempY = new double[3*m_nStaples];
  g_pdTempPhi = new double[3*m_nStaples];
  g_pdXMoved = new double[3*m_nStaples];
  g_pdYMoved = new double[3*m_nStaples];
  g_pdCOM = new double[m_nStaples];
  g_pdMOI = new double[m_nStaples];
  g_pdDtCoeffSin = new double[m_nStaples];
  g_pdDtCoeffCos = new double[m_nStaples];
  cout << "Done." << endl;
#endif
  
  cout << "...stresses" << endl;
  // Stress, energy, & force data
  cudaHostAlloc((void **)&h_pfSE, 4*sizeof(float), 0);
  m_pfEnergy = h_pfSE;
  m_pfPxx = h_pfSE+1;
  m_pfPyy = h_pfSE+2;
  m_pfPxy = h_pfSE+3;
  m_fP = 0;
  cout << "A" << endl;
  m_pfAvgAngVelo = 0.0;
  m_pnTotContacts = 0;
  cout << "B" << endl;
  cudaHostAlloc((void **)&h_pdSE, 4*sizeof(double), 0);
  m_pdEnergy = h_pdSE;
  m_pdPxx = h_pdSE+1;
  m_pdPyy = h_pdSE+2;
  m_pdPxy = h_pdSE+3;
  m_dP = 0;
  h_pdFx = new double[m_nStaples];
  h_pdFy = new double[m_nStaples];
  h_pdFt = new double[m_nStaples];
  h_pnContacts = new int[m_nStaples];
  // GPU
  cudaMalloc((void**) &d_pfSE, 4*sizeof(float));
  cudaMalloc((void**) &d_pdSE, 4*sizeof(double));
  cudaMalloc((void**) &d_pdFx, m_nStaples*sizeof(double));
  cudaMalloc((void**) &d_pdFy, m_nStaples*sizeof(double));
  cudaMalloc((void**) &d_pdFt, m_nStaples*sizeof(double));
  cudaMalloc((void**) &d_pnContacts, m_nStaples*sizeof(int));
  cudaMalloc((void**) &d_pfAvgAngVelo, sizeof(float));
  cudaMalloc((void**) &d_pnTotContacts, sizeof(int));
  m_nDeviceMem += 5*sizeof(float) + sizeof(int) + m_nStaples*(3*sizeof(double)+sizeof(int));
 #if GOLD_FUNCS == 1
  cout << "Allocating stress arrays" << endl;
  g_pfSE = new float[4];
  g_pdFx = new double[m_nStaples];
  g_pdFy = new double[m_nStaples];
  g_pdFt = new double[m_nStaples];
  cout << "Done." << endl;
#endif

  cout << "...neighbors" << endl;
  // Cell & neighbor data
  h_pnCellID = new int[3*m_nStaples];
  cudaMalloc((void**) &d_pnCellID, sizeof(int)*3*m_nStaples);
  m_nDeviceMem += 3*m_nStaples*sizeof(int);
#if GOLD_FUNCS == 1
  cout << "Allocating cell array" << endl;
  g_pnCellID = new int[3*m_nStaples];
  cout << "Done." << endl;
#endif
  cout << "Configuring cells..." << endl;
  configure_cells();
  
  h_pnNPP = new int[3*m_nStaples];
  h_pnNbrList = new int[3*m_nStaples*m_nMaxNbrs];
  cudaMalloc((void**) &d_pnNPP, sizeof(int)*3*m_nStaples);
  cudaMalloc((void**) &d_pnNbrList, sizeof(int)*3*m_nStaples*m_nMaxNbrs);
  m_nDeviceMem += 3*m_nStaples*(1+m_nMaxNbrs)*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnNPP = new int[3*m_nStaples];
  g_pnNbrList = new int[3*m_nStaples*m_nMaxNbrs];
#endif
  
  cout << "Setting up gpu kernel configurations..." << endl;
  set_kernel_configs();	
  cudaMalloc((void**) &d_pdBlockSE, 4*m_nSpGridSize*sizeof(double));
}

double Staple_Box::calculate_packing()
{
  double dParticleArea = 0.0;
  for (int p = 0; p < m_nStaples; p++)
    {
      dParticleArea += (4 * h_pdAspn[p] + 8 * h_pdAbrb[p] + 3 * D_PI * h_pdR[p]) * h_pdR[p];
    }
  return dParticleArea / (m_dL * m_dL);
}

// Creates the class
// See staple_box.h for default values of parameters
Staple_Box::Staple_Box(int nStaples, double dL, double dRMax, double dAMax, double dEpsilon, int nMaxPPC, int nMaxNbrs, Potential ePotential)
{
  assert(nStaples > 0);
  cout << "Initializing box..." << endl;
  m_nStaples = nStaples;
  assert(dL > 0.0);
  m_dL = dL;
  m_ePotential = ePotential;

  m_dEpsilon = dEpsilon;
  m_nMaxPPC = nMaxPPC;
  m_nMaxNbrs = nMaxNbrs;
  m_dRMax = dRMax;
  m_dAMax = dAMax;
  m_nDeviceMem = 0;

  cout << "Allocating memory..." << endl;
  // This allocates the coordinate data as page-locked memory, which
  //  transfers faster, since they are likely to be transferred often
  cudaHostAlloc((void**)&h_pdX, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdY, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdPhi, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdR, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAspn, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAbrb, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pnMemID, nStaples*sizeof(int), 0);
  m_dPacking = 0;
  // This initializes the arrays on the GPU
  cudaMalloc((void**) &d_pdX, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdY, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdPhi, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdR, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdAspn, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdAbrb, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pnInitID, sizeof(int)*nStaples);
  cudaMalloc((void**) &d_pnMemID, sizeof(int)*nStaples);
  // Spherocylinders
  cudaMalloc((void**) &d_pdSpX, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpY, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpPhi, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpA, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpR, sizeof(double)*3*nStaples);
  m_nDeviceMem += 23*nStaples*sizeof(double);
#if GOLD_FUNCS == 1
  cout << "Allocating staple arrays" << endl;
  g_pdX = new double[nStaples];
  g_pdY = new double[nStaples];
  g_pdPhi = new double[nStaples];
  g_pdR = new double[nStaples];
  g_pdAspn = new double[nStaples];
  g_pdAbrb = new double[nStaples];
  g_pnInitID = new int[nStaples];
  g_pnMemID = new int[nStaples];
  g_pdSpX = new double[3*nStaples];
  g_pdSpY = new double[3*nStaples];
  g_pdSpPhi = new double[3*nStaples];
  g_pdSpA = new double[3*nStaples];
  g_pdSpR = new double[3*nStaples];
  cout << "Done." << endl;
#endif

  cout << "...defaults" << endl;
  construct_defaults();
  cout << "Memory allocated on device (MB): " << (double)m_nDeviceMem / (1024.*1024.) << endl;
  place_random_staples();
  cout << "Random staples placed" << endl;
  //display(0,0,0,0);
  
}
// Create class with coordinate arrays provided
Staple_Box::Staple_Box(int nStaples, double dL, double *pdX, double *pdY, 
		       double *pdPhi, double *pdR, double *pdAspn, 
		       double *pdAbrb, double dEpsilon, int nMaxPPC, 
		       int nMaxNbrs, Potential ePotential)
{
  assert(nStaples > 0);
  m_nStaples = nStaples;
  assert(dL > 0);
  m_dL = dL;
  m_ePotential = ePotential;

  m_dEpsilon = dEpsilon;
  m_nMaxPPC = nMaxPPC;
  m_nMaxNbrs = nMaxNbrs;

  // This allocates the coordinate data as page-locked memory, which 
  //  transfers faster, since they are likely to be transferred often
  cudaHostAlloc((void**)&h_pdX, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdY, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdPhi, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdR, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAspn, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdAbrb, nStaples*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pnMemID, nStaples*sizeof(int), 0);
  m_dRMax = 0.0;
  m_dAMax = 0.0;
  for (int p = 0; p < nStaples; p++)
    {
      h_pdX[p] = pdX[p];
      h_pdY[p] = pdY[p];
      h_pdPhi[p] = pdPhi[p];
      h_pdR[p] = pdR[p];
      h_pdAspn[p] = pdAspn[p];
      h_pdAbrb[p] = pdAbrb[p];
      h_pnMemID[p] = p;
      if (pdR[p] > m_dRMax)
	m_dRMax = pdR[p];
      if (pdAbrb[p] > m_dAMax)
	m_dAMax = pdAbrb[p];
      if (pdAspn[p] > m_dAMax)
	m_dAMax = pdAspn[p];
      while (h_pdX[p] > dL)
	h_pdX[p] -= dL;
      while (h_pdX[p] < 0)
	h_pdX[p] += dL;
      while (h_pdY[p] > dL)
	h_pdY[p] -= dL;
      while (h_pdY[p] < 0)
	h_pdY[p] += dL;
    }
  m_dPacking = calculate_packing();

  // This initializes the arrays on the GPU
  cudaMalloc((void**) &d_pdX, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdY, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdPhi, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdR, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdAspn, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pdAbrb, sizeof(double)*nStaples);
  cudaMalloc((void**) &d_pnInitID, sizeof(int)*nStaples);
  cudaMalloc((void**) &d_pnMemID, sizeof(int)*nStaples);
  // This copies the values to the GPU asynchronously, which allows the
  //  CPU to go on and process further instructions while the GPU copies.
  //  Only workes on page-locked memory (allocated with cudaHostAlloc)
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdR, h_pdR, sizeof(double)*nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAspn, h_pdAspn, sizeof(double)*nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAbrb, h_pdAbrb, sizeof(double)*nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, d_pnMemID, sizeof(int)*nStaples, cudaMemcpyDeviceToDevice);
  // Spherocylinders
  cudaMalloc((void**) &d_pdSpX, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpY, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpPhi, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpA, sizeof(double)*3*nStaples);
  cudaMalloc((void**) &d_pdSpR, sizeof(double)*3*nStaples);
  m_nDeviceMem += 23*nStaples*sizeof(double);

#if GOLD_FUNCS == 1
  g_pdX = new double[nStaples];
  g_pdY = new double[nStaples];
  g_pdPhi = new double[nStaples];
  g_pdR = new double[nStaples];
  g_pdAspn = new double[nStaples];
  g_pdAbrb = new double[nStaples];
  g_pnInitID = new int[nStaples];
  g_pnMemID = new int[nStaples];
  for (int p = 0; p < nStaples; p++)
    {
      g_pdX[p] = h_pdX[p];
      g_pdY[p] = h_pdY[p];
      g_pdPhi[p] = h_pdPhi[p];
      g_pdR[p] = h_pdR[p];
      g_pdAspn[p] = h_pdAspn[p];
      g_pdAbrb[p] = h_pdAbrb[p];
      g_pnMemID[p] = h_pnMemID[p];
      g_pnInitID[p] = g_pnMemID[p];
    }
  g_pdSpX = new double[3*nStaples];
  g_pdSpY = new double[3*nStaples];
  g_pdSpPhi = new double[3*nStaples];
  g_pdSpA = new double[3*nStaples];
  g_pdSpR = new double[3*nStaples];
#endif

  construct_defaults();
  cout << "Memory allocated on device (MB): " << (double)m_nDeviceMem / (1024.*1024.) << endl;
  // Get spheocyl coordinates from staples
  split_staples();

  cudaThreadSynchronize();
  //display(0,0,0,0);
}

//Cleans up arrays when class is destroyed
Staple_Box::~Staple_Box()
{
  // Host arrays
  cudaFreeHost(h_pdX);
  cudaFreeHost(h_pdY);
  cudaFreeHost(h_pdPhi);
  cudaFreeHost(h_pdR);
  cudaFreeHost(h_pdAspn);
  cudaFreeHost(h_pdAbrb);
  cudaFreeHost(h_pnMemID);
  cudaFreeHost(h_bNewNbrs);
  cudaFreeHost(h_pfSE);
  cudaFreeHost(h_pdSE);
  delete[] h_pdFx;
  delete[] h_pdFy;
  delete[] h_pdFt;
  delete[] h_pnContacts;
  delete[] h_pnCellID;
  delete[] h_pnPPC;
  delete[] h_pnCellList;
  delete[] h_pnAdjCells;
  delete[] h_pnNPP;
  delete[] h_pnNbrList;
  delete[] h_pdMOI;
  delete[] h_pdDtCoeffSin;
  delete[] h_pdDtCoeffCos;
  
  // Device arrays
  cudaFree(d_pdX);
  cudaFree(d_pdY);
  cudaFree(d_pdPhi);
  cudaFree(d_pdR);
  cudaFree(d_pdAspn);
  cudaFree(d_pdAbrb);
  cudaFree(d_pdTempX);
  cudaFree(d_pdTempY);
  cudaFree(d_pdTempPhi);
  cudaFree(d_pnInitID);
  cudaFree(d_pnMemID);
  cudaFree(d_pdSpX);
  cudaFree(d_pdSpY);
  cudaFree(d_pdSpPhi);
  cudaFree(d_pdSpR);
  cudaFree(d_pdSpA);
  cudaFree(d_pdCOM);
  cudaFree(d_pdMOI);
  cudaFree(d_pdDtCoeffSin);
  cudaFree(d_pdDtCoeffCos);
  cudaFree(d_pdXMoved);
  cudaFree(d_pdYMoved);
  cudaFree(d_bNewNbrs);
  cudaFree(d_pfSE);
  cudaFree(d_pdFx);
  cudaFree(d_pdFy);
  cudaFree(d_pdFt);
  cudaFree(d_pnContacts);
  cudaFree(d_pnCellID);
  cudaFree(d_pnPPC);
  cudaFree(d_pnCellList);
  cudaFree(d_pnAdjCells);
  cudaFree(d_pnNPP);
  cudaFree(d_pnNbrList);
  cudaFree(d_pdBlockSE);
#if GOLD_FUNCS == 1
  delete[] g_pdX;
  delete[] g_pdY;
  delete[] g_pdPhi;
  delete[] g_pdR;
  delete[] g_pdAspn;
  delete[] g_pdAbrb;
  delete[] g_pdTempX;
  delete[] g_pdTempY;
  delete[] g_pdTempPhi;
  delete[] g_pdSpX;
  delete[] g_pdSpY;
  delete[] g_pdSpPhi;
  delete[] g_pdSpR;
  delete[] g_pdSpA;
  delete[] g_pdCOM;
  delete[] g_pdMOI;
  delete[] g_pdDtCoeffSin;
  delete[] g_pdDtCoeffCos;
  delete[] g_pdXMoved;
  delete[] g_pdYMoved;
  delete[] g_pnInitID;
  delete[] g_pnMemID;
  delete[] g_pfSE;
  delete[] g_pdFx;
  delete[] g_pdFy;
  delete[] g_pdFt;
  delete[] g_pnCellID;
  delete[] g_pnPPC;
  delete[] g_pnCellList;
  delete[] g_pnAdjCells;
  delete[] g_pnNPP;
  delete[] g_pnNbrList;
  delete g_bNewNbrs;
#endif 
}

// Display various info about the configuration which has been calculated
// Mostly used to make sure things are working right
void Staple_Box::display(bool bParticles, bool bCells, bool bNeighbors, bool bStress)
{
  double *h_pdMOI = new double[m_nStaples];
  cudaMemcpy(h_pdMOI, d_pdMOI, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  if (bParticles)
    {
      double *h_pdCOM = new double[m_nStaples];
      double *h_pdSpX = new double[m_nStaples*3];
      double *h_pdSpY = new double[m_nStaples*3];
      double *h_pdSpPhi = new double[m_nStaples*3];
      cudaMemcpyAsync(h_pdX, d_pdX, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdR, d_pdR, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdAspn, d_pdAspn, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdAbrb, d_pdAbrb, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pnMemID, d_pnMemID, sizeof(int)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdCOM, d_pdCOM, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
      checkCudaError("Display: copying particle data to host");
      
      cout << endl << "Box dimension: " << m_dL << endl;
      cout << "Gamma: " << m_dGamma << endl;
      for (int p = 0; p < m_nStaples; p++)
	{
	  int m = h_pnMemID[p];
	  cout << "Particle " << p << " (" << m  << "): (" << h_pdX[m] << ", " 
	       << h_pdY[m] << ", " << h_pdPhi[m] << ") R = " << h_pdR[m] 
	       << " Aspine = " << h_pdAspn[m] << " Abarb = " << h_pdAbrb[m]
	       << " MOI = " << h_pdMOI[m] << " COM = " << h_pdCOM[m] << endl;
	}

      cout << endl;
      cudaMemcpy(h_pdSpX, d_pdSpX, sizeof(double)*3*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdSpY, d_pdSpY, sizeof(double)*3*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdSpPhi, d_pdSpPhi, sizeof(double)*3*m_nStaples, cudaMemcpyDeviceToHost);
      for (int sp = 0; sp < 3*m_nStaples; sp++) {
	cout << "Spherocylinder " << sp << ": " << h_pdSpX[sp] << " " 
	     << h_pdSpY[sp] << " " << h_pdSpPhi[sp] << endl; 
      }
      delete h_pdCOM;
      delete[] h_pdSpX; delete[] h_pdSpY; delete[] h_pdSpPhi;
    }
  if (bCells)
    {
      cudaMemcpy(h_pnPPC, d_pnPPC, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost); 
      cudaMemcpy(h_pnCellList, d_pnCellList, sizeof(int)*m_nCells*m_nMaxPPC, cudaMemcpyDeviceToHost);
      checkCudaError("Display: copying cell data to host");

      cout << endl;
      int nTotal = 0;
      int nMaxPPC = 0;
      for (int c = 0; c < m_nCells; c++)
	{
	  nTotal += h_pnPPC[c];
	  nMaxPPC = max(nMaxPPC, h_pnPPC[c]);
	  cout << "Cell " << c << ": " << h_pnPPC[c] << " particles\n";
	  for (int p = 0; p < h_pnPPC[c]; p++)
	    {
	      cout << h_pnCellList[c*m_nMaxPPC + p] << " ";
	    }
	  cout << endl;
	}
      cout << "Total particles in cells: " << nTotal << endl;
      cout << "Maximum particles in any cell: " << nMaxPPC << endl;
    }
  if (bNeighbors)
    {
      cudaMemcpy(h_pnNPP, d_pnNPP, sizeof(int)*3*m_nStaples, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*3*m_nStaples*m_nMaxNbrs, cudaMemcpyDeviceToHost);
      checkCudaError("Display: copying neighbor data to host");

      cout << endl;
      int nMaxNPP = 0;
      for (int p = 0; p < 3*m_nStaples; p++)
	{
	  nMaxNPP = max(nMaxNPP, h_pnNPP[p]);
	  cout << "Particle " << p << ": " << h_pnNPP[p] << " neighbors\n";
	  for (int n = 0; n < h_pnNPP[p]; n++)
	    {
	      cout << h_pnNbrList[n*3*m_nStaples + p] << " ";
	    }
	  cout << endl;
	}
      cout << "Maximum neighbors of any particle: " << nMaxNPP << endl;
    }
  if (bStress)
    {
      cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFx, d_pdFx, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFy, d_pdFy, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFt, d_pdFt, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pnContacts, d_pnContacts, m_nStaples*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&m_pnTotContacts, d_pnTotContacts, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&m_pfAvgAngVelo, d_pfAvgAngVelo, sizeof(float), cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      cout << endl;
      double dAV = 0.0;
      for (int p = 0; p < m_nStaples; p++)
	{
	  cout << "Particle " << p << ":  (" << h_pdFx[p] << ", " << h_pdFy[p] << ", " 
	       << h_pdFt[p] << ") " << h_pnContacts[p] << "\n";
	  dAV += h_pdFt[p] / h_pdMOI[p];
	}
      dAV /= m_nStaples;
      cout << endl << "Total contacts: " << m_pnTotContacts << endl;
      cout << "Average angular velocity: " << m_pfAvgAngVelo << " " << dAV << endl;
      cout << "Energy: " << *m_pfEnergy << endl;
      cout << "Pxx: " << *m_pfPxx << endl;
      cout << "Pyy: " << *m_pfPyy << endl;
      cout << "Total P: " << m_fP << endl;
      cout << "Pxy: " << *m_pfPxy << endl;
    }
  delete[] h_pdMOI; 
}

bool Staple_Box::check_for_contacts(int nIndex)
{
  double dC = 2*m_dAMax*m_dAMax / (3*m_dAMax + 2*m_dRMax);
  double dS = 4*( (m_dAMax + 3*m_dRMax)*(m_dAMax + 3*m_dRMax) + 
		  (2*m_dAMax + m_dRMax - dC)*(2*m_dAMax + m_dRMax - dC) );

  double dStX = h_pdX[nIndex];
  double dStY = h_pdY[nIndex];
  for (int p = 0; p < nIndex; p++) {
    //cout << "Checking: " << nIndex << " vs " << p << endl;
    double dStXj = h_pdX[p];
    double dStYj = h_pdY[p];

    double dDelX = dStX - dStXj;
    double dDelY = dStY - dStYj;
    dDelX += m_dL * ((dDelX < -0.5*m_dL) - (dDelX > 0.5*m_dL));
    dDelY += m_dL * ((dDelY < -0.5*m_dL) - (dDelY > 0.5*m_dL));
    dDelX += m_dGamma * dDelY;
    double dDelRSqr = dDelX * dDelX + dDelY * dDelY;
    if (dDelRSqr < dS) {
      double pdX[2] = {dStX, dStXj};
      double pdY[2] = {dStY, dStYj};
      double pdPhi[2] = {h_pdPhi[nIndex], h_pdPhi[p]};
      double pdR[2] = {h_pdR[nIndex], h_pdR[p]};
      double pdAs[2] = {h_pdAspn[nIndex], h_pdAspn[p]};
      double pdAb[2] = {h_pdAbrb[nIndex], h_pdAbrb[p]};
      double pdSpX[6];
      double pdSpY[6];
      double pdSpPhi[6];
      double pdSpR[6];
      double pdSpA[6];
      
      for (int thid = 0; thid < 2; thid++) {
	double dX = pdX[thid];
	double dY = pdY[thid];
	double dPhi = pdPhi[thid];
	double dR = pdR[thid];
	double dAs = pdAs[thid];
	double dAb = pdAb[thid];
	double dCOM = 2*dAb*dAb / (dAs + 2*dR + 2*dAb);

	// Coordinates of spine
	double dDeltaY = dCOM * cos(dPhi);
	pdSpY[3*thid] = dY + dDeltaY;
	pdSpX[3*thid] = dX - dCOM * sin(dPhi) - m_dGamma * dDeltaY;
	pdSpPhi[3*thid] = dPhi;
	pdSpR[3*thid] = dR;
	pdSpA[3*thid] = dAs;

	// Coordinates of barbs
	dDeltaY = (dAs + 2.*dR) * sin(dPhi)
	  - (dAb - dCOM) * cos(dPhi);
	pdSpY[3*thid + 1] = dY + dDeltaY;
	pdSpX[3*thid + 1] = dX + (dAs + 2.*dR) * cos(dPhi) 
	  + (dAb - dCOM) * sin(dPhi) - m_dGamma * dDeltaY;
	pdSpPhi[3*thid + 1] = dPhi + 0.5 * D_PI;
	pdSpR[3*thid + 1] = dR;
	pdSpA[3*thid + 1] = dAb;
	
	dDeltaY = -(dAs + 2.*dR) * sin(dPhi)
	  - (dAb - dCOM) * cos(dPhi);
	pdSpY[3*thid + 2] = dY + dDeltaY;
	pdSpX[3*thid + 2] = dX - (dAs + 2.*dR) * cos(dPhi) 
	  + (dAb - dCOM) * sin(dPhi) - m_dGamma * dDeltaY;
	pdSpPhi[3*thid + 2] = dPhi + 0.5 * D_PI;
	pdSpR[3*thid + 2] = dR;
	pdSpA[3*thid + 2] = dAb;
      }
      
      for (int nPID = 0; nPID < 3; nPID++) {
	double dX = pdSpX[nPID];
	double dY = pdSpY[nPID];
	double dPhi = pdSpPhi[nPID];
	double dR = pdSpR[nPID];
	double dA = pdSpA[nPID];
    
	for (int nAdjPID = 3; nAdjPID < 6; nAdjPID++) {
	  double dDeltaX = dX - pdSpX[nAdjPID];
	  double dDeltaY = dY - pdSpY[nAdjPID];
	  double dPhiB = pdSpPhi[nAdjPID];
	  double dSigma = dR + pdSpR[nAdjPID];
	  double dB = pdSpA[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += m_dL * ((dDeltaX < -0.5*m_dL) - (dDeltaX > 0.5*m_dL));
	  dDeltaY += m_dL * ((dDeltaY < -0.5*m_dL) - (dDeltaY > 0.5*m_dL));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += m_dGamma * dDeltaX;

	  double nxA = dA * cos(dPhi);
	  double nyA = dA * sin(dPhi);
	  double nxB = dB * cos(dPhiB);
	  double nyB = dB * sin(dPhiB);
	  
	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  
	  double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	  double s = -(b*t+d)/a;
	  double sarg = fabs(s);
	  s = fmin( fmax(s,-1.), 1. );
	  if (sarg > 1) 
	    t = fmin( fmax( -(b*s+e)/a, -1.), 1.);
	  
	  // Check if they overlap and calculate forces
	  double dDx = dDeltaX + s*nxA - t*nxB;
	  double dDy = dDeltaY + s*nyA - t*nyB;
	  double dDSqr = dDx * dDx + dDy * dDy;
	  if (dDSqr < dSigma*dSigma || dDSqr != dDSqr)
	    return 1;
	}
      }
    }
  }
  return 0;
}

void Staple_Box::place_random_staples(int seed)
{
  srand(time(0) + seed);

  for (int p = 0; p < m_nStaples; p++) {
    h_pdR[p] = m_dRMax;
    h_pdAspn[p] = m_dAMax;
    h_pdAbrb[p] = m_dAMax;
    h_pnMemID[p] = p;
  }
  cudaMemcpyAsync(d_pdR, h_pdR, sizeof(double)*m_nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAspn, h_pdAspn, sizeof(double)*m_nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdAbrb, h_pdAbrb, sizeof(double)*m_nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, h_pnMemID, sizeof(int)*m_nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*m_nStaples, cudaMemcpyHostToDevice);
#if GOLD_FUNCS == 1
  for (int p = 0; p < m_nStaples; p++) {
    g_pdR[p] = h_pdR[p];
    g_pdAspn[p] = h_pdAspn[p];
    g_pdAbrb[p] = h_pdAbrb[p];
    g_pnInitID[p] = h_pnMemID[p];
    g_pnMemID[p] = h_pnMemID[p];
  }
#endif
  cudaThreadSynchronize();

  h_pdX[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  h_pdY[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  h_pdPhi[0] = 2*D_PI * static_cast<double>(rand())/static_cast<double>(RAND_MAX);

  for (int p = 1; p < m_nStaples; p++) {
    bool bContact = 1;
    int nTries = 0;

    while (bContact) {
      h_pdX[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      h_pdY[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      h_pdPhi[p] = 2*D_PI * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

      bContact = check_for_contacts(p);
      nTries += 1;
    }
    cout << "Staple " << p << " placed in " << nTries << " attempts." << endl;
  }
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*m_nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*m_nStaples, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*m_nStaples, cudaMemcpyHostToDevice);
#if GOLD_FUNCS == 1
  for (int p = 0; p < m_nStaples; p++) {
    g_pdX[p] = h_pdX[p];
    g_pdY[p] = h_pdY[p];
    g_pdPhi[p] = h_pdPhi[p];
  }
#endif
  cudaThreadSynchronize();
  cout << "Data copied to device" << endl;

  split_staples();
  cout << "Staples split into constituent spherocylinders" << endl;
}
