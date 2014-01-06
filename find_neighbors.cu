// -*- c++ -*-
/*
* find_neighbors.cu
*
*  
*
*
*/

#include <cuda.h>
#include "staple_box.h"
#include "cudaErr.h"
#include "data_primitives.h"
#include <math.h>
#include <limits>

using namespace std;

const double D_PI = 3.14159265358979;
const double D_MIN = numeric_limits<double>::min();


__global__ void find_com(int nStaples, double *pdCOM, double *pdR, double *pdAs, double *pdAb)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;

  while (thid < nStaples) {
    double dR = pdR[thid];
    double dAs = pdAs[thid];
    double dAb = pdAb[thid];

    pdCOM[thid] = (8.*dAb*dAb + 2.*D_PI*dAb*dR) / (8.*dAb + 4.*dAs + 3.*D_PI*dR);
    
    thid += blockDim.x * gridDim.x;
  }
}

__global__ void calc_rot_consts(int nStaples, double *pdR, double *pdAs, 
			       double *pdAb, double *pdCOM, double *pdMOI, 
			       double *pdSinCoeff, double *pdCosCoeff)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nStaples) {
    double dR = pdR[nPID];
    double dAs = pdAs[nPID];
    double dA = dAs + 2. * dR;
    //double dA = 3.;
    double dB = pdAb[nPID];
    //double dB = 2.;
    double dC = 2*dB*dB / (dA + 2*dB);
    pdCOM[nPID] = dC;

    double dIntdS = 2*dA + 4*dB;
    double dIntSy2SinCoeff = dA*dA*(2*dA/3 + 4*dB);
    double dIntSy2CosCoeff = dB*dB*(16*dB/3 - 8*dC) + 2*(dA + 2*dB)*dC*dC;
    double dIntS2 = dIntSy2SinCoeff + dIntSy2CosCoeff;
    pdMOI[nPID] = dIntS2 / dIntdS;
    pdSinCoeff[nPID] = dIntSy2SinCoeff / dIntS2;
    pdCosCoeff[nPID] = dIntSy2CosCoeff / dIntS2;

    nPID += nThreads;
  }
  
}

// Treating staple as 3 rods rather than one bent rod
__global__ void calc_rot_consts2(int nStaples, double *pdR, double *pdAs, 
				 double *pdAb, double *pdCOM, double *pdMOI, 
				 double *pdSinCoeff, double *pdCosCoeff)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nStaples) {
    double dR = pdR[nPID];
    double dA = pdAs[nPID];
    double dAR = dA + 2. * dR;
    //double dA = 3.;
    double dB = pdAb[nPID];
    //double dB = 2.;
    double dC = 2*dB*dB / (dA + 2*dB);
    pdCOM[nPID] = dC;

    double dIntdS = 2*dA + 4*dB;
    double dIntSy2SinCoeff = (2*dA*dA*dA/3 + 4*dAR*dAR*dB);
    double dIntSy2CosCoeff = dB*dB*(16*dB/3 - 8*dC) + 2*(dA + 2*dB)*dC*dC;
    double dIntS2 = dIntSy2SinCoeff + dIntSy2CosCoeff;
    pdMOI[nPID] = dIntS2 / dIntdS;
    pdSinCoeff[nPID] = dIntSy2SinCoeff / dIntS2;
    pdCosCoeff[nPID] = dIntSy2CosCoeff / dIntS2;

    nPID += nThreads;
  }
  
}

__global__ void split_staple_2_sph(int nStaples, double dGamma, double *pdX, 
				   double *pdY, double *pdPhi, double *pdR, 
				   double *pdAs, double *pdAb, double *pdCOM, 
				   double *pdSpX,double *pdSpY,double *pdSpPhi,
				   double *pdSpR, double *pdSpA)
{
  int thid = blockDim.x * blockIdx.x + threadIdx.x;
  
  while (thid < nStaples) {
    double dX = pdX[thid];
    double dY = pdY[thid];
    double dPhi = pdPhi[thid];
    double dR = pdR[thid];
    double dAs = pdAs[thid];
    double dAb = pdAb[thid];
    double dCOM = pdCOM[thid];

    // Coordinates of spine
    double dDeltaY = dCOM * cos(dPhi);
    pdSpY[3*thid] = dY + dDeltaY;
    pdSpX[3*thid] = dX - dCOM * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid] = dPhi;
    pdSpR[3*thid] = dR;
    pdSpA[3*thid] = dAs;

    // Coordinates of barbs
    dDeltaY = (dAs + 2.*dR) * sin(dPhi)
      - (dAb - dCOM) * cos(dPhi);
    pdSpY[3*thid + 1] = dY + dDeltaY;
    pdSpX[3*thid + 1] = dX + (dAs + 2.*dR) * cos(dPhi) 
      + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid + 1] = dPhi + 0.5 * D_PI;
    pdSpR[3*thid + 1] = dR;
    pdSpA[3*thid + 1] = dAb;

    dDeltaY = -(dAs + 2.*dR) * sin(dPhi)
      - (dAb - dCOM) * cos(dPhi);
    pdSpY[3*thid + 2] = dY + dDeltaY;
    pdSpX[3*thid + 2] = dX - (dAs + 2.*dR) * cos(dPhi) 
      + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid + 2] = dPhi + 0.5 * D_PI;
    pdSpR[3*thid + 2] = dR;
    pdSpA[3*thid + 2] = dAb;

    thid += blockDim.x * gridDim.x;
  }
}


__global__ void constrain_coords(int nStaples, double dL, double *pdX, double *pdY, double *pdPhi)
{
  int nID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  
  while (nID < nStaples) {
    double dX = pdX[nID];
    double dY = pdY[nID];
    double dPhi = pdPhi[nID];

    if (dY > dL)
      {
	dY -= dL;
	pdY[nID] = dY;
      }
    else if (dY < 0)
      {
	dY += dL;
	pdY[nID] = dY;
      }
    if (dX > dL)
      {
	dX -= dL;
	pdX[nID] = dX;
      }
    else if (dX < 0)
      {
	dX += dL;
	pdX[nID] = dX;
      }
    if (dPhi < -D_PI)
      {
	dPhi += 2*D_PI;
	pdPhi[nID] = dPhi;
      }
    else if (dPhi > D_PI)
      {
	dPhi -= 2*D_PI;
	pdPhi[nID] = dPhi;
      }

    nID += nThreads;
  }
}


///////////////////////////////////////////////////////////////
// Find the Cell ID for each particle:
//  The list of cell IDs for each particle is returned to pnCellID
//  A list of which particles are in each cell is returned to pnCellList
//
// *NOTE* if there are more than nMaxPPC particles in a given cell,
//  not all of these particles will get added to the cell list
///////////////////////////////////////////////////////////////
__global__ void find_cells(int nStaples, int nMaxPPC, double dCellW, double dCellH,
			   int nCellCols, double dL, double *pdX, double *pdY, 
			   int *pnCellID, int *pnPPC, int *pnCellList)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < 3*nStaples) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    
    // I often allow the stored coordinates to drift slightly outside the box limits
    //  until 
    if (dY > dL)
      {
	dY -= dL;
	pdY[nPID] = dY;
      }
    else if (dY < 0)
      {
	dY += dL;
	pdY[nPID] = dY;
      }
    if (dX > dL)
      {
	dX -= dL;
	pdX[nPID] = dX;
      }
    else if (dX < 0)
      {
	dX += dL;
	pdX[nPID] = dX;
      }

    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    pnCellID[nPID] = nCellID;

    // Add 1 particle to a cell safely (only allows one thread to access the memory
    //  address at a time). nPPC is the original value, not the result of addition 
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    // only add particle to cell if there is not already the maximum number in cell
    if (nPPC < nMaxPPC)
      pnCellList[nCellID * nMaxPPC + nPPC] = nPID;
    else
      nPPC = atomicAdd(pnPPC + nCellID, -1);

    nPID += nThreads;
  }
}


////////////////////////////////////////////////////////////////
// Here a list of possible contacts is created for each spherocylinder
//  The list of neighbors is returned to pnNbrList
//
////////////////////////////////////////////////////////////////
__global__ void find_nbrs(int nStaples, int nMaxPPC, int *pnCellID, int *pnPPC, 
			  int *pnCellList, int *pnAdjCells, int nMaxNbrs, int *pnNPP, 
			  int *pnNbrList, double *pdX, double *pdY, double *pdR, 
			  double *pdA, double dEpsilon, double dL, double dGamma)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = gridDim.x * blockDim.x;

  while (nPID < 3*nStaples)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      int nNbrs = 0;

      // Particles in adjacent cells are added if they are close enough to 
      //  interact without each moving by more than dEpsilon/2
      int nCellID = pnCellID[nPID];
      int nP = pnPPC[nCellID];
      for (int p = 0; p < nP; p++)
	{
	  int nAdjPID = pnCellList[nCellID*nMaxPPC + p];
	  if (nAdjPID / 3 != nPID / 3)
	    {
	      double dSigma = dR + dA + pdR[nAdjPID] + pdA[nAdjPID] + dEpsilon;
	      double dDeltaY = dY - pdY[nAdjPID];
	      dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
	      
	      if (fabs(dDeltaY) < dSigma)
		{
		  double dDeltaX = dX - pdX[nAdjPID];
		  dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
		  double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		  double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		  if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
		    {
		      if (nNbrs < nMaxNbrs)
			{
			  pnNbrList[3*nStaples * nNbrs + nPID] = nAdjPID; // This indexing makes for more coalesced reads from global memory
			  nNbrs += 1;
			}
		    }
		}
	    }
	}

      for (int nc = 0; nc < 8; nc++)
	{
	  int nAdjCID = pnAdjCells[8 * nCellID + nc];
	  nP = pnPPC[nAdjCID];
	  for (int p = 0; p < nP; p++)
	    {
	      int nAdjPID = pnCellList[nAdjCID*nMaxPPC + p];
	      if (nAdjPID / 3 != nPID / 3) {
		// The maximum distance at which two particles could contact
		//  plus a little bit of moving room - dEpsilon 
		double dSigma = dR + dA + pdA[nAdjPID] + pdR[nAdjPID] + dEpsilon;
		double dDeltaY = dY - pdY[nAdjPID];
		
		// Make sure were finding the closest separation
		dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
		
		if (fabs(dDeltaY) < dSigma)
		  {
		    double dDeltaX = dX - pdX[nAdjPID];
		    dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
		    
		    // Go to unsheared coordinates
		    double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		    // Also look at distance when the strain parameter is at its max (0.5)
		    double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		    if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
		      {
			if (nNbrs < nMaxNbrs)
			  {
			    pnNbrList[3*nStaples * nNbrs + nPID] = nAdjPID;
			    nNbrs += 1;
			  }
		      }
		  }
	      }
	    } 
	}
      
      pnNPP[nPID] = nNbrs;
      nPID += nThreads;
    }
}


void Staple_Box::split_staples()
{
  if (!m_bCOM) {
    calc_rot_consts <<<m_nGridSize, m_nBlockSize>>> 
      (m_nStaples, d_pdR, d_pdAspn, d_pdAbrb, d_pdCOM, 
       d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos);
    cudaThreadSynchronize();
    checkCudaError("Finding center of mass and rotational constants"); 
    //printf("Centers of mass found\n");
    m_bCOM = 1;

    /*
    double *h_pdCOM = (double*) malloc(sizeof(double)*m_nStaples);
    double *h_pdMOI = (double*) malloc(sizeof(double)*m_nStaples);
    double *h_pdDtCoeffSin = (double*) malloc(sizeof(double)*m_nStaples);
    double *h_pdDtCoeffCos = (double*) malloc(sizeof(double)*m_nStaples);
    cudaMemcpy(h_pdCOM, d_pdCOM, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    */
    cudaMemcpy(h_pdMOI, d_pdMOI, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pdDtCoeffSin, d_pdDtCoeffSin, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pdDtCoeffCos, d_pdDtCoeffCos, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    /*
    for (int s = 0; s < m_nStaples; s++)
      printf("C: %f, I: %f, A: %f, B: %f\n", h_pdCOM[s], h_pdMOI[s], h_pdDtCoeffSin[s], h_pdDtCoeffCos[s]);
    free(h_pdCOM); free(h_pdMOI); free(h_pdDtCoeffSin); free(h_pdDtCoeffCos);
    */
  }

  split_staple_2_sph <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, m_dGamma, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAspn, 
     d_pdAbrb, d_pdCOM, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA);
  cudaThreadSynchronize();
  checkCudaError("Splitting staples to spherocylinder coordinates");

  /*
  printf("Staples split into constituent spherocylinders\n");
  fflush(stdout);

  double *h_pdSpX = (double*) malloc(sizeof(double)*3*m_nStaples); 
  double *h_pdSpY = (double*) malloc(sizeof(double)*3*m_nStaples);
  double *h_pdSpPhi = (double*) malloc(sizeof(double)*3*m_nStaples);
  cudaMemcpy(h_pdSpX, d_pdSpX, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpY, d_pdSpY, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpPhi, d_pdSpPhi, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);

  for (int s = 0; s < m_nStaples; s++) {
    printf("Staple %d: %g, %g, %g\n", s, h_pdX[s], h_pdY[s], h_pdPhi[s]);
    printf("%g, %g, %g\n", h_pdSpX[3*s], h_pdSpY[3*s], h_pdSpPhi[3*s]);
    printf("%g, %g, %g\n", h_pdSpX[3*s+1], h_pdSpY[3*s+1], h_pdSpPhi[3*s+1]);
    printf("%g, %g, %g\n", h_pdSpX[3*s+2], h_pdSpY[3*s+2], h_pdSpPhi[3*s+2]);
    fflush(stdout);
  }

  free(h_pdSpX); free(h_pdSpY); free(h_pdSpPhi);
  */
}

///////////////////////////////////////////////////////////////
// Finds a list of possible contacts for each particle
//
// Usually when things are moving I keep track of an Xmoved and Ymoved
//  and only call this to make a new list of neighbors if some particle
//  has moved more than (dEpsilon / 2) in some direction
//////////////////////////////////////////////////////////////
void Staple_Box::find_neighbors()
{
  // reset each byte to 0
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);
  cudaMemset((void *) d_pnNPP, 0, sizeof(int)*3*m_nStaples);
  cudaMemset((void *) d_pdXMoved, 0, sizeof(double)*m_nStaples);
  cudaMemset((void *) d_pdYMoved, 0, sizeof(double)*m_nStaples);
  cudaMemset((void *) d_bNewNbrs, 0, sizeof(int));

  constrain_coords <<<m_nGridSize, m_nBlockSize>>>
    (m_nStaples, m_dL, d_pdX, d_pdY, d_pdPhi);

  find_cells <<<m_nSpGridSize, m_nSpBlockSize>>>
    (m_nStaples, m_nMaxPPC, m_dCellW, m_dCellH, m_nCellCols, 
     m_dL, d_pdSpX, d_pdSpY, d_pnCellID, d_pnPPC, d_pnCellList);
  cudaThreadSynchronize();
  checkCudaError("Finding cells");


  find_nbrs <<<m_nSpGridSize, m_nSpBlockSize>>>
    (m_nStaples, m_nMaxPPC, d_pnCellID, d_pnPPC, d_pnCellList, d_pnAdjCells, 
     m_nMaxNbrs, d_pnNPP, d_pnNbrList, d_pdSpX, d_pdSpY, d_pdSpR, d_pdSpA, 
     m_dEpsilon, m_dL, m_dGamma);
  cudaThreadSynchronize();
  checkCudaError("Finding neighbors");

  /*
  int *h_pnCellID = (int*) malloc(sizeof(int)*3*m_nStaples);
  int *h_pnNPP = (int*) malloc(sizeof(int)*3*m_nStaples);
  int *h_pnNbrList = (int*) malloc(sizeof(int)*3*m_nStaples*m_nMaxNbrs);
  cudaMemcpy(h_pnCellID, d_pnCellID, sizeof(int)*3*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNPP,d_pnNPP, sizeof(int)*3*m_nStaples,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*3*m_nStaples*m_nMaxNbrs, cudaMemcpyDeviceToHost);

  for (int p = 0; p < 3*m_nStaples; p++) {
    printf("Spherocyl: %d, Cell: %d, neighbors: %d\n", 
	   p, h_pnCellID[p], h_pnNPP[p]);
    for (int n = 0; n < h_pnNPP[p]; n++) {
      printf("%d ", h_pnNbrList[n*3*m_nStaples + p]);
    }
    printf("\n");
    fflush(stdout);
  }

  free(h_pnCellID); free(h_pnNPP); free(h_pnNbrList);
  */
}


__global__ void find_nbrs_2(int nStaples, int nMaxNbrs, int *pnNPP, int *pnNbrList, double *pdX, double *pdY, 
			    double *pdR, double *pdA, double dEpsilon, double dL, double dGamma)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = gridDim.x * blockDim.x;

  while (nPID < 3*nStaples) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    double dR = pdR[nPID];
    double dA = pdA[nPID];
    int nAdjID = threadIdx.y;
    while (nAdjID < 3*nStaples)
      {
	// Particles in adjacent cells are added if they are close enough to 
	//  interact without each moving by more than dEpsilon/
	if (nAdjID / 3 != nPID / 3)
	  {
	    double dSigma = dR + dA + pdR[nAdjID] + pdA[nAdjID] + dEpsilon;
	    double dDeltaY = dY - pdY[nAdjID];
	    dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
	    
	    if (fabs(dDeltaY) < dSigma)
	      {
		double dDeltaX = dX - pdX[nAdjID];
		dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
		double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
		  {
		    // This indexing makes global memory accesses more coalesced
		    int nNbrs = atomicAdd(pnNPP + nPID, 1);
		    if (nNbrs < nMaxNbrs)
		      {
			pnNbrList[3*nStaples * nNbrs + nPID] = nAdjID;
			nNbrs += 1;
		      }
		    else
		      nNbrs = atomicAdd(pnNPP + nPID, -1);
		  }
	      }
	  }
	nAdjID += blockDim.y;
      }
    nPID += nThreads;
  }
}

void Staple_Box::find_neighbors_2()
{
  cudaMemset((void *) d_pnNPP, 0, sizeof(int)*3*m_nStaples);
  cudaMemset((void *) d_pdXMoved, 0, sizeof(double)*m_nStaples);
  cudaMemset((void *) d_pdYMoved, 0, sizeof(double)*m_nStaples);
  cudaMemset((void *) d_bNewNbrs, 0, sizeof(int));

  int nBlockDimX = 48;
  int nBlockDimY = 16;
  int nGridDimX = 3*m_nStaples / nBlockDimX;
  dim3 gridDim(nGridDimX);
  dim3 blockDim(nBlockDimX, nBlockDimY);

  constrain_coords <<<m_nGridSize, m_nBlockSize>>>
    (m_nStaples, m_dL, d_pdX, d_pdY, d_pdPhi);

  find_nbrs_2 <<<gridDim, blockDim>>>
    (m_nStaples, m_nMaxNbrs, d_pnNPP, d_pnNbrList, d_pdSpX, d_pdSpY, d_pdSpR, d_pdSpA, m_dEpsilon, m_dL, m_dGamma);
  cudaThreadSynchronize();
  checkCudaError("Finding neighbors (2)");
}


////////////////////////////////////////////////////////////////////////////////////
// Sets gamma back by 1 (used when gamma > 0.5)
//  also finds the cells in the process
//
///////////////////////////////////////////////////////////////////////////////////
__global__ void set_back_coords(int nStaples, double dL, double *pdX, double *pdY, double *pdPhi)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nStaples) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    double dPhi = pdPhi[nPID];
    
    // I often allow the stored coordinates to drift slightly outside the box limits
    if (dPhi > D_PI)
      {
	dPhi -= 2*D_PI;
	pdPhi[nPID] = dPhi;
      }
    else if (dPhi < -D_PI)
      {
	dPhi += 2*D_PI;
	pdPhi[nPID] = dPhi;
      }
    if (dY > dL)
      {
	dY -= dL;
	pdY[nPID] = dY;
      }
    else if (dY < 0)
      {
	dY += dL;
	pdY[nPID] = dY;
      }
    
    // When gamma -> gamma-1, Xi -> Xi + Yi
    dX += dY;
    if (dX < 0)
      {
	dX += dL;
      }
    while (dX > dL)
      {
	dX -= dL;
      }
    pdX[nPID] = dX;

    nPID += nThreads;
  }
  

}

void Staple_Box::set_back_gamma()
{
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);
  cudaMemset((void *) d_pdXMoved, 0, sizeof(double)*m_nStaples);
  cudaMemset((void *) d_pdYMoved, 0, sizeof(double)*m_nStaples);
  cudaMemset((void *) d_bNewNbrs, 0, sizeof(int));

  set_back_coords <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, m_dL, d_pdX, d_pdY, d_pdPhi);
  cudaThreadSynchronize();
  checkCudaError("Finding new coordinates, cells");
  m_dGamma -= 1;

  split_staples();
  find_neighbors();
}


////////////////////////////////////////////////////////////////////////////
// Finds cells for all particles regardless of maximum particle per cell
//  used for reordering particles
/////////////////////////////////////////////////////////////////////////
__global__ void find_cells_nomax(int nStaples, double dCellW, double dCellH,
				 int nCellCols, double dL, double *pdX, double *pdY, 
				 int *pnCellID, int *pnPPC)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nStaples) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    
    // Particles are allowed to drift slightly outside the box limits
    //  until cells are reassigned due to a particle drift of dEpsilon/2 
    if (dY > dL) {
      dY -= dL; 
      pdY[nPID] = dY; }
    else if (dY < 0) {
      dY += dL;
      pdY[nPID] = dY; }
    if (dX > dL) {
      dX -= dL; 
      pdX[nPID] = dX; }
    else if (dX < 0) {
      dX += dL;
      pdX[nPID] = dX; }

    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    
    pnCellID[nPID] = nCellID;
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    nPID += nThreads; }
}

__global__ void reorder_part(int nStaples, double *pdTempX, double *pdTempY, double *pdTempPhi,
			     double *pdTempR, double *pdTempAs, double *pdTempAb, int *pnTempInitID, 
			     double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdAs, 
			     double *pdAb, int *pnInitID, int *pnMemID, int *pnCellID, int *pnCellSID)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nStaples) {
    double dX = pdTempX[nPID];
    double dY = pdTempY[nPID];
    double dPhi = pdTempPhi[nPID];
    double dR = pdTempR[nPID];
    double dAs = pdTempAs[nPID];
    double dAb = pdTempAb[nPID];
    int nInitID = pnTempInitID[nPID];

    int nCellID = pnCellID[nPID];
    int nNewID = atomicAdd(pnCellSID + nCellID, 1);
    
    pdX[nNewID] = dX;
    pdY[nNewID] = dY;
    pdPhi[nNewID] = dPhi;
    pdR[nNewID] = dR;
    if (pdR[nNewID] < 1e-16)  // Not sure why, but some radii were evaluating to 0 here
      pdR[nNewID] = 0.5;
    pdAs[nNewID] = dAs;
    pdAb[nNewID] = dAb;
    pnMemID[nInitID] = nNewID;
    pnInitID[nNewID] = nInitID;

    nPID += nThreads; 
  }
}

__global__ void invert_IDs(int nIDs, int *pnIn, int *pnOut)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (thid < nIDs) {
    int i = pnIn[thid];
    pnOut[i] = thid; 
    thid += nThreads; }
    
}

void Staple_Box::reorder_particles()
{
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);

  //find particle cell IDs and number of particles in each cell
  find_cells_nomax <<<m_nGridSize, m_nBlockSize>>>
    (m_nStaples, m_dCellW, m_dCellH, m_nCellCols, 
     m_dL, d_pdX, d_pdY, d_pnCellID, d_pnPPC);
  cudaThreadSynchronize();
  checkCudaError("Reordering particles: Finding cells");

  int *d_pnCellSID;
  int *d_pnTempInitID;
  double *d_pdTempR; 
  double *d_pdTempAs; 
  double *d_pdTempAb;
  cudaMalloc((void **) &d_pnCellSID, sizeof(int) * m_nCells);
  cudaMalloc((void **) &d_pdTempR, sizeof(double) * m_nStaples);
  cudaMalloc((void **) &d_pdTempAs, sizeof(double) * m_nStaples);
  cudaMalloc((void **) &d_pdTempAb, sizeof(double) * m_nStaples);
  cudaMalloc((void **) &d_pnTempInitID, sizeof(int) * m_nStaples);
  cudaMemcpy(d_pdTempX, d_pdX, sizeof(double) * m_nStaples, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempY, d_pdY, sizeof(double) * m_nStaples, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempPhi, d_pdPhi, sizeof(double) * m_nStaples, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempR, d_pdR, sizeof(double) * m_nStaples, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempAs, d_pdAspn, sizeof(double) * m_nStaples, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempAb, d_pdAbrb, sizeof(double) * m_nStaples, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pnTempInitID, d_pnInitID, sizeof(int) * m_nStaples, cudaMemcpyDeviceToDevice);

  exclusive_scan(d_pnPPC, d_pnCellSID, m_nCells);

  /*
  int *h_pnCellSID = (int*) malloc(m_nCells * sizeof(int));
  int *h_pnCellNPart = (int*) malloc(m_nCells * sizeof(int));
  cudaMemcpy(h_pnCellNPart, d_pnCellNPart, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnCellSID, d_pnCellSID, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  for (int c = 0; c < m_nCells; c++)
    {
      printf("%d %d\n", h_pnCellNPart[c], h_pnCellSID[c]);
    }
  free(h_pnCellSID);
  free(h_pnCellNPart);
  */

  //reorder particles based on cell ID (first by Y direction)
  reorder_part <<<m_nGridSize, m_nBlockSize>>>
    (m_nStaples, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdTempR, d_pdTempAs, d_pdTempAb, 
     d_pnTempInitID, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAspn, d_pdAbrb, d_pnInitID, 
     d_pnMemID, d_pnCellID, d_pnCellSID);
  cudaThreadSynchronize();
  checkCudaError("Reordering particles: changing order");

  //invert_IDs <<<m_nGridSize, m_nBlockSize>>> (m_nStaples, d_pnMemID, d_pnInitID);
  cudaMemcpyAsync(h_pnMemID, d_pnMemID, m_nStaples*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdR, d_pdR, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdAspn, d_pdAspn, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdAbrb, d_pdAbrb, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  cudaFree(d_pnCellSID); cudaFree(d_pnTempInitID);
  cudaFree(d_pdTempR); cudaFree(d_pdTempAs); cudaFree(d_pdTempAb);

  m_bCOM = 0;
  split_staples();
  find_neighbors();
}


////////////////////////////////////////////////////////////////////////
// Sets the particle IDs to their order in memory
//  so the current IDs become the initial IDs
/////////////////////////////////////////////////////////////////////
void Staple_Box::reset_IDs()
{
  ordered_array(d_pnInitID, m_nStaples, m_nGridSize, m_nBlockSize);
  cudaMemcpyAsync(h_pnMemID, d_pnInitID, sizeof(int)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_pnMemID, d_pnInitID, sizeof(int)*m_nStaples, cudaMemcpyDeviceToDevice);
  
}



















/******************************************************************************
 'gold' functions for debugging
 ******************************************************************************/

#if GOLD_FUNCS == 1

void calc_rot_consts_gold(int nGridDim, int nBlockDim, int nStaples, double *pdR, 
			  double *pdAs, double *pdAb, double *pdCOM, double *pdMOI, 
			  double *pdSinCoeff, double *pdCosCoeff)
{
  for (int blockIdx = 0; blockIdx < nGridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < nBlockDim; threadIdx++) {
      int nPID = threadIdx + blockIdx * nBlockDim;
      int nThreads = nBlockDim * nGridDim;

      while (nPID < nStaples) {
	double dR = pdR[nPID];
	double dAs = pdAs[nPID];
	double dA = dAs + 2. * dR;
	//double dA = 3.;
	double dB = pdAb[nPID];
	//double dB = 2.;
	double dC = 2*dB*dB / (dA + 2*dB);
	pdCOM[nPID] = dC;
	
	double dIntdS = 2*dA + 4*dB;
	double dIntSy2SinCoeff = dA*dA*(2*dA/3 + 4*dB);
	double dIntSy2CosCoeff = dB*dB*(16*dB/3 - 8*dC) + 2*(dA + 2*dB)*dC*dC;
	double dIntS2 = dIntSy2SinCoeff + dIntSy2CosCoeff;
	pdMOI[nPID] = dIntS2 / dIntdS;
	pdSinCoeff[nPID] = dIntSy2SinCoeff / dIntS2;
	pdCosCoeff[nPID] = dIntSy2CosCoeff / dIntS2;
	//printf("bid: %d, tid: %d, pid:%d, COM: %.4g, MOI: %.4g, C1: %.4g, C2 %.4g\n",
	//       blockIdx, threadIdx, nPID, pdCOM[nPID], pdMOI[nPID], pdSinCoeff[nPID], pdCosCoeff[nPID]);
	
	nPID += nThreads;
      }
    }
  }
  
}

void split_staple_2_sph_gold(int gridDim, int blockDim, int nStaples, double dGamma, double *pdX, double *pdY, 
			     double *pdPhi, double *pdR, double *pdAs, double *pdAb, double *pdCOM, 
			     double *pdSpX,double *pdSpY,double *pdSpPhi, double *pdSpR, double *pdSpA)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      int thid = blockDim * blockIdx + threadIdx;
  
      while (thid < nStaples) {
	double dX = pdX[thid];
	double dY = pdY[thid];
	double dPhi = pdPhi[thid];
	double dR = pdR[thid];
	double dAs = pdAs[thid];
	double dAb = pdAb[thid];
	double dCOM = pdCOM[thid];
	
	// Coordinates of spine
	double dDeltaY = dCOM * cos(dPhi);
	pdSpY[3*thid] = dY + dDeltaY;
	pdSpX[3*thid] = dX - dCOM * sin(dPhi) - dGamma * dDeltaY;
	pdSpPhi[3*thid] = dPhi;
	pdSpR[3*thid] = dR;
	pdSpA[3*thid] = dAs;
	
	// Coordinates of barbs
	dDeltaY = (dAs + 2.*dR) * sin(dPhi)
	  - (dAb - dCOM) * cos(dPhi);
	pdSpY[3*thid + 1] = dY + dDeltaY;
	pdSpX[3*thid + 1] = dX + (dAs + 2.*dR) * cos(dPhi) 
	  + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
	pdSpPhi[3*thid + 1] = dPhi + 0.5 * D_PI;
	pdSpR[3*thid + 1] = dR;
	pdSpA[3*thid + 1] = dAb;
	
	dDeltaY = -(dAs + 2.*dR) * sin(dPhi)
	  - (dAb - dCOM) * cos(dPhi);
	pdSpY[3*thid + 2] = dY + dDeltaY;
	pdSpX[3*thid + 2] = dX - (dAs + 2.*dR) * cos(dPhi) 
	  + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
	pdSpPhi[3*thid + 2] = dPhi + 0.5 * D_PI;
	pdSpR[3*thid + 2] = dR;
	pdSpA[3*thid + 2] = dAb;
	//printf("bid: %d,tid: %d, pid:%d, X: %.4g, Y: %.4g, Phi: %.4g\n",
	//       blockIdx, threadIdx, thid, pdX[thid], pdY[thid], pdPhi[thid]);
        //printf("Spine: (%.4g, %.4g, %.4g), Barbs: (%.4g, %.4g, %.4g), (%.4g, %.4g, %.4g)\n",
	//       pdSpX[3*thid], pdSpY[3*thid], pdSpPhi[3*thid], pdSpX[3*thid+1], pdSpY[3*thid+1], 
	//       pdSpPhi[3*thid+1], pdSpX[3*thid+2], pdSpY[3*thid+2], pdSpPhi[3*thid+2]); 
	
	thid += blockDim * gridDim;
      }
    }
  }
}

void Staple_Box::split_staples_gold()
{
  m_bCOM = 0;
  if (!m_bCOM) {
    calc_rot_consts_gold(m_nGridSize, m_nBlockSize, m_nStaples, g_pdR, g_pdAspn, g_pdAbrb, 
			 g_pdCOM, g_pdMOI, g_pdDtCoeffSin, g_pdDtCoeffCos);
    m_bCOM = 1;

    /*
    double *h_pdCOM = (double*) malloc(sizeof(double)*m_nStaples);
    double *h_pdMOI = (double*) malloc(sizeof(double)*m_nStaples);
    double *h_pdDtCoeffSin = (double*) malloc(sizeof(double)*m_nStaples);
    double *h_pdDtCoeffCos = (double*) malloc(sizeof(double)*m_nStaples);
    cudaMemcpy(h_pdCOM, d_pdCOM, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pdMOI, d_pdMOI, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pdDtCoeffSin, d_pdDtCoeffSin, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pdDtCoeffCos, d_pdDtCoeffCos, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    for (int s = 0; s < m_nStaples; s++)
      printf("C: %f, I: %f, A: %f, B: %f\n", h_pdCOM[s], h_pdMOI[s], h_pdDtCoeffSin[s], h_pdDtCoeffCos[s]);
    free(h_pdCOM); free(h_pdMOI); free(h_pdDtCoeffSin); free(h_pdDtCoeffCos);
    */
  }

  split_staple_2_sph_gold(m_nGridSize, m_nBlockSize, m_nStaples, m_dGamma, g_pdX, g_pdY, g_pdPhi, g_pdR, 
			  g_pdAspn, g_pdAbrb, g_pdCOM, g_pdSpX, g_pdSpY, g_pdSpPhi, g_pdSpR, g_pdSpA);
  /*
  printf("Staples split into constituent spherocylinders\n");
  fflush(stdout);
  */
}

void Staple_Box::compare_split_staples()
{
  split_staples();
  double *h_pdSpX = (double*) malloc(sizeof(double)*3*m_nStaples); 
  double *h_pdSpY = (double*) malloc(sizeof(double)*3*m_nStaples);
  double *h_pdSpPhi = (double*) malloc(sizeof(double)*3*m_nStaples);
  cudaMemcpy(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdPhi, d_pdPhi, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpX, d_pdSpX, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpY, d_pdSpY, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpPhi, d_pdSpPhi, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);

  split_staples_gold();

  double dTol = 0.000001;
  for (int s = 0; s < m_nStaples; s++) {
    if (fabs(h_pdX[s]) < fabs(g_pdX[s]) * (1. - dTol) || fabs(h_pdX[s]) > fabs(g_pdX[s]) * (1. + dTol) ||
	fabs(h_pdY[s]) < fabs(g_pdY[s]) * (1. - dTol) || fabs(h_pdY[s]) > fabs(g_pdY[s]) * (1. + dTol) ||
	fabs(h_pdPhi[s]) < fabs(g_pdPhi[s]) * (1. - dTol) || fabs(h_pdPhi[s]) > fabs(g_pdPhi[s]) * (1. + dTol) ) {
      printf("Staple coordinate difference detected for staple %d:\n", s);
      printf("GPU: %.9g, %.9g, %.9g\n", h_pdX[s], h_pdY[s], h_pdPhi[s]);
      printf("Gld: %.9g, %.9g, %.9g\n", g_pdX[s], g_pdY[s], g_pdPhi[s]);
    }
    for (int p = 0; p < 3; p++) {
      if (fabs(h_pdSpX[3*s + p]) < fabs(g_pdSpX[3*s + p]) * (1. - dTol) || 
	  fabs(h_pdSpX[3*s + p]) > fabs(g_pdSpX[3*s + p]) * (1. + dTol) ||
	  fabs(h_pdSpY[3*s + p]) < fabs(g_pdSpY[3*s + p]) * (1. - dTol) || 
	  fabs(h_pdSpY[3*s + p]) > fabs(g_pdSpY[3*s + p]) * (1. + dTol) ||
	  fabs(h_pdSpPhi[3*s+p]) < fabs(g_pdSpPhi[3*s+p]) * (1. - dTol) || 
	  fabs(h_pdSpPhi[3*s+p]) > fabs(g_pdSpPhi[3*s+p]) * (1. + dTol) ) {
	printf("Staple coordinate difference detected for staple %d, spherocyl %d:\n", s, p);
	printf("GPU: %.9g, %.9g, %.9g\n", h_pdSpX[3*s+p], h_pdSpY[3*s+p], h_pdSpPhi[3*s+p]);
	printf("Gld: %.9g, %.9g, %.9g\n", g_pdSpX[3*s+p], g_pdSpY[3*s+p], g_pdSpPhi[3*s+p]);	
      }
    }
  }
  free(h_pdSpX); free(h_pdSpY); free(h_pdSpPhi);
}


void constrain_coords_gold(int gridDim, int blockDim, int nStaples, double dL, double *pdX, double *pdY, double *pdPhi)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      int nID = threadIdx + blockIdx * blockDim;
      int nThreads = blockDim * gridDim;
  
      while (nID < nStaples) {
	double dX = pdX[nID];
	double dY = pdY[nID];
	double dPhi = pdPhi[nID];
	
	if (dY > dL)
	  {
	    dY -= dL;
	    pdY[nID] = dY;
	  }
	else if (dY < 0)
	  {
	    dY += dL;
	    pdY[nID] = dY;
	  }
	if (dX > dL)
	  {
	    dX -= dL;
	    pdX[nID] = dX;
	  }
	else if (dX < 0)
	  {
	    dX += dL;
	    pdX[nID] = dX;
	  }
	if (dPhi < -D_PI)
	  {
	    dPhi += 2*D_PI;
	    pdPhi[nID] = dPhi;
	  }
	else if (dPhi > D_PI)
	  {
	    dPhi -= 2*D_PI;
	    pdPhi[nID] = dPhi;
	  }
	
	nID += nThreads;
      }
    }
  }
}

void find_cells_gold(int gridDim, int blockDim, int nStaples, int nMaxPPC, double dCellW, double dCellH,
		     int nCellCols, double dL, double *pdX, double *pdY, int *pnCellID, int *pnPPC, int *pnCellList)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
      int nPID = threadIdx + blockIdx * blockDim;
      int nThreads = blockDim * gridDim;

      while (nPID < 3*nStaples) {
	double dX = pdX[nPID];
	double dY = pdY[nPID];
	
	// I often allow the stored coordinates to drift slightly outside the box limits
	//  until 
	if (dY > dL)
	  {
	    dY -= dL;
	    pdY[nPID] = dY;
	  }
	else if (dY < 0)
	  {
	    dY += dL;
	    pdY[nPID] = dY;
	  }
	if (dX > dL)
	  {
	    dX -= dL;
	    pdX[nPID] = dX;
	  }
	else if (dX < 0)
	  {
	    dX += dL;
	    pdX[nPID] = dX;
	  }
	
	//find the cell ID, add a particle to that cell 
	int nCol = (int)(dX / dCellW);
	int nRow = (int)(dY / dCellH); 
	int nCellID = nCol + nRow * nCellCols;
	pnCellID[nPID] = nCellID;
	
	// Add 1 particle to a cell safely (only allows one thread to access the memory
	//  address at a time). nPPC is the original value, not the result of addition 
	int nPPC = pnPPC[nCellID];
	pnPPC[nCellID] += 1;
    
	// only add particle to cell if there is not already the maximum number in cell
	if (nPPC < nMaxPPC)
	  pnCellList[nCellID * nMaxPPC + nPPC] = nPID;
	else
	  pnPPC[nCellID] -= 1;
	
	//printf("bid: %d,tid: %d, pid:%d, X: %.4g, Y: %.4g, Cell: %d, PPC: %d\n",
	//       blockIdx, threadIdx, nPID, pdX[nPID], pdY[nPID], nCellID, pnPPC[nCellID]);
	
	nPID += nThreads;
      }
    }
  }
}

void find_nbrs_gold(int gridDim, int blockDim, int nStaples, int nMaxPPC, int *pnCellID, int *pnPPC, 
		    int *pnCellList, int *pnAdjCells, int nMaxNbrs, int *pnNPP, 
		    int *pnNbrList, double *pdX, double *pdY, double *pdR, 
		    double *pdA, double dEpsilon, double dL, double dGamma)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      int nPID = threadIdx + blockIdx * blockDim;
      int nThreads = gridDim * blockDim;

      while (nPID < 3*nStaples)
	{
	  double dX = pdX[nPID];
	  double dY = pdY[nPID];
	  double dR = pdR[nPID];
	  double dA = pdA[nPID];
	  int nNbrs = 0;

	  // Particles in adjacent cells are added if they are close enough to 
	  //  interact without each moving by more than dEpsilon/2
	  int nCellID = pnCellID[nPID];
	  int nP = pnPPC[nCellID];
	  for (int p = 0; p < nP; p++)
	    {
	      int nAdjPID = pnCellList[nCellID*nMaxPPC + p];
	      if (nAdjPID / 3 != nPID / 3)
		{
		  double dSigma = dR + dA + pdR[nAdjPID] + pdA[nAdjPID] + dEpsilon;
		  double dDeltaY = dY - pdY[nAdjPID];
		  dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
		  
		  if (fabs(dDeltaY) < dSigma)
		    {
		      double dDeltaX = dX - pdX[nAdjPID];
		      dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
		      double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		      double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		      if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
			{
			  // This indexing makes global memory accesses more coalesced
			  if (nNbrs < nMaxNbrs)
			    {
			      pnNbrList[3*nStaples * nNbrs + nPID] = nAdjPID;
			      nNbrs += 1;
			    }
			}
		    }
		}
	    }
	  
	  for (int nc = 0; nc < 8; nc++)
	    {
	      int nAdjCID = pnAdjCells[8 * nCellID + nc];
	      nP = pnPPC[nAdjCID];
	      for (int p = 0; p < nP; p++)
		{
		  int nAdjPID = pnCellList[nAdjCID*nMaxPPC + p];
		  if (nAdjPID / 3 != nPID / 3) {
		    // The maximum distance at which two particles could contact
		    //  plus a little bit of moving room - dEpsilon 
		    double dSigma = dR + dA + pdA[nAdjPID] + pdR[nAdjPID] + dEpsilon;
		    double dDeltaY = dY - pdY[nAdjPID];
		    
		    // Make sure were finding the closest separation
		    dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
		    
		    if (fabs(dDeltaY) < dSigma)
		      {
			double dDeltaX = dX - pdX[nAdjPID];
			dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
			
			// Go to unsheared coordinates
			double dDeltaRx = dDeltaX + dGamma * dDeltaY;
			// Also look at distance when the strain parameter is at its max (0.5)
			double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
			if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
			  {
			    if (nNbrs < nMaxNbrs)
			      {
				pnNbrList[3*nStaples * nNbrs + nPID] = nAdjPID;
				nNbrs += 1;
			      }
			  }
		      }
		  }
		} 
	    }
	  //printf("bid: %d,tid: %d, pid:%d, Cell: %d, NPP: %d\n",
	  //	 blockIdx, threadIdx, nPID, nCellID, nNbrs);
	  //for (int nb = 0; nb < nNbrs; nb++)
	  //  printf("%d ", pnNbrList[3*nStaples*nb + nPID]);
	  //printf("\n");
	  
	  pnNPP[nPID] = nNbrs;
	  nPID += nThreads;
	}
    }
  }
}

void Staple_Box::find_neighbors_gold()
{
  // reset each byte to 0
  for (int c = 0; c < m_nCells; c++) {
    g_pnPPC[c] = 0;
  }
  for (int p = 0; p < m_nStaples; p++) {
    g_pdXMoved[p] = 0;
    g_pdYMoved[p] = 0;
  }
  *g_bNewNbrs = 0;

  constrain_coords_gold(m_nGridSize, m_nBlockSize, m_nStaples, m_dL, g_pdX, g_pdY, g_pdPhi);

  find_cells_gold(m_nSpGridSize, m_nSpBlockSize, m_nStaples, m_nMaxPPC, m_dCellW, m_dCellH, 
		  m_nCellCols, m_dL, g_pdSpX, g_pdSpY, g_pnCellID, g_pnPPC, g_pnCellList);

  find_nbrs_gold(m_nSpGridSize, m_nSpBlockSize, m_nStaples, m_nMaxPPC, g_pnCellID, g_pnPPC, 
		 g_pnCellList, g_pnAdjCells, m_nMaxNbrs, g_pnNPP, g_pnNbrList, g_pdSpX, g_pdSpY, 
		 g_pdSpR, g_pdSpA, m_dEpsilon, m_dL, m_dGamma);
}

void Staple_Box::compare_find_neighbors()
{
  find_neighbors();
  int *h_pnCellID = (int*) malloc(sizeof(int)*3*m_nStaples);
  int *h_pnNPP = (int*) malloc(sizeof(int)*3*m_nStaples);
  int *h_pnNbrList = (int*) malloc(sizeof(int)*3*m_nStaples*m_nMaxNbrs);
  cudaMemcpy(h_pnCellID, d_pnCellID, sizeof(int)*3*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNPP,d_pnNPP, sizeof(int)*3*m_nStaples,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*3*m_nStaples*m_nMaxNbrs, cudaMemcpyDeviceToHost);

  find_neighbors_gold();
  
  for (int p = 0; p < 3*m_nStaples; p++) {
    if (h_pnCellID[p] != g_pnCellID[p] || h_pnNPP[p] != g_pnNPP[p])
      printf("Different neighbor info for staple %d, spherocyl %d\n", p / 3, p % 3);
    for (int n = 0; n < h_pnNPP[p]; n++) {
      int nid = h_pnNbrList[3*m_nStaples*n + p];
      bool found = 0;
      for (int m = 0; m < g_pnNPP[p]; m++) {
	if (g_pnNbrList[3*m_nStaples*m + p] == nid) {
	  found = 1;
	  break;
	}
      }
      if (!found) {
	printf("Particle %d neighbor %d found in gpu but not gold\n", p, nid);
      }
    }
    for (int n = 0; n < g_pnNPP[p]; n++) {
      int nid = g_pnNbrList[3*m_nStaples*n + p];
      bool found = 0;
      for (int m = 0; m < h_pnNPP[p]; m++) {
	if (h_pnNbrList[3*m_nStaples*m + p] == nid) {
	  found = 1;
	  break;
	}
      }
      if (!found) {
	printf("Particle %d neighbor %d found in gold but not gpu\n", p, nid);
      }
    } 
  }

  free(h_pnCellID); free(h_pnNPP); free(h_pnNbrList);
}


void set_back_coords_gold(int gridDim, int blockDim, int nStaples, double dL, double *pdX, double *pdY, double *pdPhi)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < blockIdx; threadIdx++) {
      // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
      int nPID = threadIdx + blockIdx * blockDim;
      int nThreads = blockDim * gridDim;

      while (nPID < nStaples) {
	double dX = pdX[nPID];
	double dY = pdY[nPID];
	double dPhi = pdPhi[nPID];
	
	// I often allow the stored coordinates to drift slightly outside the box limits
	if (dPhi > D_PI)
	  {
	    dPhi -= 2*D_PI;
	    pdPhi[nPID] = dPhi;
	  }
	else if (dPhi < -D_PI)
	  {
	    dPhi += 2*D_PI;
	    pdPhi[nPID] = dPhi;
	  }
	if (dY > dL)
	  {
	    dY -= dL;
	    pdY[nPID] = dY;
	  }
	else if (dY < 0)
	  {
	    dY += dL;
	    pdY[nPID] = dY;
	  }
	
	// When gamma -> gamma-1, Xi -> Xi + Yi
	dX += dY;
	if (dX < 0)
	  {
	    dX += dL;
	  }
	while (dX > dL)
	  {
	    dX -= dL;
	  }
	pdX[nPID] = dX;
	
	nPID += nThreads;
      }
    }
  }
}

void Staple_Box::set_back_gamma_gold()
{
  // reset each byte to 0
  for (int c = 0; c < m_nCells; c++) {
    g_pnPPC[c] = 0;
  }
  for (int p = 0; p < m_nStaples; p++) {
    g_pdXMoved[p] = 0;
    g_pdYMoved[p] = 0;
  }
  *g_bNewNbrs = 0;

  set_back_coords_gold(m_nGridSize, m_nBlockSize, m_nStaples, m_dL, g_pdX, g_pdY, g_pdPhi);

  m_dGamma -= 1;

  split_staples_gold();
  find_neighbors_gold();
}



void find_cells_nomax_gold(int gridDim, int blockDim, int nStaples, double dCellW, double dCellH,
			   int nCellCols, double dL, double *pdX, double *pdY, int *pnCellID, int *pnPPC)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
      int nPID = threadIdx + blockIdx * blockDim;
      int nThreads = blockDim * gridDim;

      while (nPID < nStaples) {
	double dX = pdX[nPID];
	double dY = pdY[nPID];
	
	// Particles are allowed to drift slightly outside the box limits
	//  until cells are reassigned due to a particle drift of dEpsilon/2 
	if (dY > dL) {
	  dY -= dL; 
	  pdY[nPID] = dY; }
	else if (dY < 0) {
	  dY += dL;
	  pdY[nPID] = dY; }
	if (dX > dL) {
	  dX -= dL; 
	  pdX[nPID] = dX; }
	else if (dX < 0) {
	  dX += dL;
	  pdX[nPID] = dX; }
	
	//find the cell ID, add a particle to that cell 
	int nCol = (int)(dX / dCellW);
	int nRow = (int)(dY / dCellH); 
	int nCellID = nCol + nRow * nCellCols;
	
	pnCellID[nPID] = nCellID;
	pnPPC[nCellID] += 1;
	printf("bid: %d,tid: %d, pid:%d, X: %.4g, Y: %.4g, Cell: %d, PPC: %d\n",
	       blockIdx, threadIdx, nPID, pdX[nPID], pdY[nPID], nCellID, pnPPC[nCellID]);
	
	nPID += nThreads; 
      }
    }
  }
}

void reorder_part_gold(int gridDim, int blockDim, int nStaples, double *pdTempX, double *pdTempY, double *pdTempPhi,
		       double *pdTempR, double *pdTempAs, double *pdTempAb, int *pnTempInitID, 
		       double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdAs, double *pdAb, 
		       int *pnInitID, int *pnMemID, int *pnCellID, int *pnCellSID)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      int nPID = threadIdx + blockIdx * blockDim;
      int nThreads = blockDim * gridDim;
      
      while (nPID < nStaples) {
	double dX = pdTempX[nPID];
	double dY = pdTempY[nPID];
	double dR = pdTempR[nPID];
	double dPhi = pdTempPhi[nPID];
	double dAs = pdTempAs[nPID];
	double dAb = pdTempAb[nPID];
	int nInitID = pnTempInitID[nPID];
	
	int nCellID = pnCellID[nPID];
	int nNewID = pnCellSID[nCellID];
	pnCellSID[nCellID] += 1;
	
	pdX[nNewID] = dX;
	pdY[nNewID] = dY;
	pdPhi[nNewID] = dPhi;
	pdR[nNewID] = dR;
	pdAs[nNewID] = dAs;
	pdAb[nNewID] = dAb;
	pnMemID[nInitID] = nNewID;
	pnInitID[nNewID] = nInitID;
	
	printf("bid: %d,tid: %d, old:%d, new: %d, init: %d, Cell: %d\n", 
	       blockIdx, threadIdx, nPID, nNewID, nInitID, nCellID);
	printf("X: %.4g, Y: %.4g, R: %.4g, As: %.4g, Ab: %.4g\n",
	       pdX[nPID], pdY[nPID], pdR[nPID], pdAs[nPID], pdAb[nPID]);
	
	nPID += nThreads; 
      }
    }
  }
}

void Staple_Box::reorder_particles_gold()
{
  for (int c = 0; c < m_nCells; c++) {
    g_pnPPC[c] = 0;
  }

  //find particle cell IDs and number of particles in each cell
  find_cells_nomax_gold(m_nGridSize, m_nBlockSize, m_nStaples, m_dCellW, m_dCellH, m_nCellCols, 
			m_dL, g_pdX, g_pdY, g_pnCellID, g_pnPPC);

  int *g_pnCellSID = (int*) malloc(m_nCells*sizeof(int));
  int *g_pnTempInitID = (int*) malloc(m_nStaples*sizeof(int));
  double *g_pdTempR = (double*) malloc(m_nStaples*sizeof(double)); 
  double *g_pdTempAs = (double*) malloc(m_nStaples*sizeof(double)); 
  double *g_pdTempAb = (double*) malloc(m_nStaples*sizeof(double));
  for (int s = 0; s < m_nStaples; s++) {
    g_pdTempX[s] = g_pdX[s];
    g_pdTempY[s] = g_pdY[s];
    g_pdTempPhi[s] = g_pdPhi[s];
    g_pdTempR[s] = g_pdR[s];
    g_pdTempAs[s] = g_pdAspn[s];
    g_pdTempAb[s] = g_pdAbrb[s];    
    g_pnTempInitID[s] = g_pnInitID[s];
  }

  g_pnCellSID[0] = 0;
  for (int c = 1; c < m_nCells; c++) {
    g_pnCellSID[c] = g_pnCellSID[c-1] + g_pnPPC[c-1];
  }

  /*
  int *h_pnCellSID = (int*) malloc(m_nCells * sizeof(int));
  int *h_pnCellNPart = (int*) malloc(m_nCells * sizeof(int));
  cudaMemcpy(h_pnCellNPart, d_pnCellNPart, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnCellSID, d_pnCellSID, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  for (int c = 0; c < m_nCells; c++)
    {
      printf("%d %d\n", h_pnCellNPart[c], h_pnCellSID[c]);
    }
  free(h_pnCellSID);
  free(h_pnCellNPart);
  */

  //reorder particles based on cell ID (first by Y direction)
  reorder_part_gold(m_nGridSize, m_nBlockSize, m_nStaples, g_pdTempX, g_pdTempY, g_pdTempPhi, 
		    g_pdTempR, g_pdTempAs, g_pdTempAb, g_pnTempInitID, g_pdX, g_pdY, g_pdPhi, 
		    g_pdR, g_pdAspn, g_pdAbrb, g_pnInitID, g_pnMemID, g_pnCellID, g_pnCellSID);


  free(g_pnCellSID); free(g_pnTempInitID);
  free(g_pdTempR); free(g_pdTempAs); free(g_pdTempAb);

  m_bCOM = 0;
  split_staples_gold();
  find_neighbors_gold();
}

void Staple_Box::compare_reorder_particles()
{
  reorder_particles();
  double *h_pdSpX = (double*) malloc(sizeof(double)*3*m_nStaples); 
  double *h_pdSpY = (double*) malloc(sizeof(double)*3*m_nStaples);
  double *h_pdSpPhi = (double*) malloc(sizeof(double)*3*m_nStaples);
  double *h_pdSpR = (double*) malloc(sizeof(double)*3*m_nStaples); 
  double *h_pdSpA = (double*) malloc(sizeof(double)*3*m_nStaples);
  cudaMemcpy(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdPhi, d_pdPhi, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpX, d_pdSpX, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpY, d_pdSpY, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpPhi, d_pdSpPhi, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpR, d_pdSpR, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpA, d_pdSpA, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  int *h_pnCellID = (int*) malloc(sizeof(int)*3*m_nStaples);
  int *h_pnNPP = (int*) malloc(sizeof(int)*3*m_nStaples);
  int *h_pnNbrList = (int*) malloc(sizeof(int)*3*m_nStaples*m_nMaxNbrs);
  cudaMemcpy(h_pnCellID, d_pnCellID, sizeof(int)*3*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNPP,d_pnNPP, sizeof(int)*3*m_nStaples,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*3*m_nStaples*m_nMaxNbrs, cudaMemcpyDeviceToHost);
 
  reorder_particles_gold();
 
  double dTol = 0.0000001;
  for (int s = 0; s < m_nStaples; s++) {
    if (h_pdR[s] < g_pdR[s] * (1. - dTol) || h_pdR[s] > g_pdR[s] * (1. + dTol) ||
	h_pdAspn[s] < g_pdAspn[s] * (1. - dTol) || h_pdAspn[s] > g_pdAspn[s] * (1. + dTol) ||
	h_pdAbrb[s] < g_pdAbrb[s] * (1. - dTol) || h_pdAbrb[s] > g_pdAbrb[s] * (1. + dTol) ) {
      printf("Staple constant difference detected for staple %d:\n", s);
      printf("GPU: %.9g, %.9g, %.9g\n", h_pdR[s], h_pdAspn[s], h_pdAbrb[s]);
      printf("Gld: %.9g, %.9g, %.9g\n", g_pdR[s], g_pdAspn[s], g_pdAbrb[s]);
    }
    for (int p = 0; p < 3; p++) {
      if (h_pdSpR[3*s + p] < g_pdSpR[3*s + p] * (1. - dTol) || h_pdSpR[3*s + p] > g_pdSpR[3*s + p] * (1. + dTol) ||
	  h_pdSpA[3*s + p] < g_pdSpA[3*s + p] * (1. - dTol) || h_pdSpA[3*s + p] > g_pdSpA[3*s + p] * (1. + dTol) ) {
	printf("Spherocyl constant difference detected for staple %d, spherocyl %d:\n", s, p);
	printf("GPU: %.9g, %.9g\n", h_pdSpR[3*s+p], h_pdSpA[3*s+p]);
	printf("Gld: %.9g, %.9g\n", g_pdSpR[3*s+p], g_pdSpA[3*s+p]);	
      }
    }
  }
  for (int s = 0; s < m_nStaples; s++) {
    if (fabs(h_pdX[s]) < fabs(g_pdX[s]) * (1. - dTol) || fabs(h_pdX[s]) > fabs(g_pdX[s]) * (1. + dTol) ||
	fabs(h_pdY[s]) < fabs(g_pdY[s]) * (1. - dTol) || fabs(h_pdY[s]) > fabs(g_pdY[s]) * (1. + dTol) ||
	fabs(h_pdPhi[s]) < fabs(g_pdPhi[s]) * (1. - dTol) || fabs(h_pdPhi[s]) > fabs(g_pdPhi[s]) * (1. + dTol) ) {
      g_pdTempX[s] = g_pdX[s];
      g_pdTempY[s] = g_pdY[s];
      g_pdTempPhi[s] = g_pdPhi[s];
      g_pdX[s] = g_pdX[s+1];
      g_pdY[s] = g_pdY[s+1];
      g_pdPhi[s] = g_pdPhi[s+1];
      g_pdX[s+1] = g_pdTempX[s];
      g_pdY[s+1] = g_pdTempY[s];
      g_pdPhi[s+1] = g_pdTempPhi[s];
      split_staples_gold();
      if (fabs(h_pdX[s]) < fabs(g_pdX[s]) * (1. - dTol) || fabs(h_pdX[s]) > fabs(g_pdX[s]) * (1. + dTol) ||
	  fabs(h_pdY[s]) < fabs(g_pdY[s]) * (1. - dTol) || fabs(h_pdY[s]) > fabs(g_pdY[s]) * (1. + dTol) ||
	  fabs(h_pdPhi[s]) < fabs(g_pdPhi[s]) * (1. - dTol) || fabs(h_pdPhi[s]) > fabs(g_pdPhi[s]) * (1. + dTol) ) {
	printf("Staple coordinate difference detected for staple %d:\n", s);
	printf("GPU: %.9g, %.9g, %.9g\n", h_pdX[s], h_pdY[s], h_pdPhi[s]);
	printf("Gld: %.9g, %.9g, %.9g\n", g_pdX[s], g_pdY[s], g_pdPhi[s]);
      }
    }
    for (int p = 0; p < 3; p++) {
      if (fabs(h_pdSpX[3*s + p]) < fabs(g_pdSpX[3*s + p]) * (1. - dTol) || 
	  fabs(h_pdSpX[3*s + p]) > fabs(g_pdSpX[3*s + p]) * (1. + dTol) ||
	  fabs(h_pdSpY[3*s + p]) < fabs(g_pdSpY[3*s + p]) * (1. - dTol) || 
	  fabs(h_pdSpY[3*s + p]) > fabs(g_pdSpY[3*s + p]) * (1. + dTol) ||
	  fabs(h_pdSpPhi[3*s+p]) < fabs(g_pdSpPhi[3*s+p]) * (1. - dTol) || 
	  fabs(h_pdSpPhi[3*s+p]) > fabs(g_pdSpPhi[3*s+p]) * (1. + dTol) ) {
	printf("Spherocyl coordinate difference detected for staple %d, spherocyl %d:\n", s, p);
	printf("GPU: %.9g, %.9g, %.9g\n", h_pdSpX[3*s+p], h_pdSpY[3*s+p], h_pdSpPhi[3*s+p]);
	printf("Gld: %.9g, %.9g, %.9g\n", g_pdSpX[3*s+p], g_pdSpY[3*s+p], g_pdSpPhi[3*s+p]);	
      }
    }
  }
  free(h_pdSpX); free(h_pdSpY); free(h_pdSpPhi);
  find_neighbors_gold();
  for (int p = 0; p < 3*m_nStaples; p++) {
    if (h_pnCellID[p] != g_pnCellID[p] || h_pnNPP[p] != g_pnNPP[p])
      printf("Different neighbor info for staple %d, spherocyl %d\n", p / 3, p % 3);
    for (int n = 0; n < h_pnNPP[p]; n++) {
      int nid = h_pnNbrList[3*m_nStaples*n + p];
      bool found = 0;
      for (int m = 0; m < g_pnNPP[p]; m++) {
	if (g_pnNbrList[3*m_nStaples*m + p] == nid) {
	  found = 1;
	  break;
	}
      }
      if (!found) {
	printf("Particle %d neighbor %d found in gpu but not gold\n", p, nid);
      }
    }
    for (int n = 0; n < g_pnNPP[p]; n++) {
      int nid = g_pnNbrList[3*m_nStaples*n + p];
      bool found = 0;
      for (int m = 0; m < h_pnNPP[p]; m++) {
	if (h_pnNbrList[3*m_nStaples*m + p] == nid) {
	  found = 1;
	  break;
	}
      }
      if (!found) {
	printf("Particle %d neighbor %d found in gold but not gpu\n", p, nid);
      }
    } 
  }
  free(h_pnCellID); free(h_pnNPP); free(h_pnNbrList);

}


#endif
