// -*- c++ -*-
#include <math.h>
#include <stdlib.h>
#include <time.h>


// This kernel calculates the center of mass, moment of inertia, and the coefficents 
__global__ void calc_rotational_consts(int nStaples, double *pdR, double *pdAs, double *pdAb, 
		    double *pdCOM, double *pdMOI, double *pdSinCoeff, double *pdCosCoeff)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nStaples) {
    double dR = pdR[nPID];
    double dAs = pdAs[nPID];
    double dA = dAs + 2. * dR;
    double dB = pdAb[nPID];
    double dC = 2*dB*dB / (dA + 2*dB);
    pdCOM[nPID] = dC;  // Center of mass of staple

    double dIntdS = 2*dA + 4*dB;
    double dIntSy2SinCoeff = dA*dA*(2*dA/3 + 4*dB);
    double dIntSy2CosCoeff = dB*dB*(16*dB/3 - 8*dC) + 2*(dA + 2*dB)*dC*dC;
    double dIntS2 = dIntSy2SinCoeff + dIntSy2CosCoeff;
    pdMOI[nPID] = dIntS2 / dIntdS;  // Moment of inertia (for going from torque to rotation)
    // The rotation due to the background shear velocity is A*Sin(t)^2+B*Cos(t)^2,
    // These are the coefficients A and B
    pdSinCoeff[nPID] = dIntSy2SinCoeff / dIntS2;
    pdCosCoeff[nPID] = dIntSy2CosCoeff / dIntS2;

    nPID += nThreads;
  }
  
}

// From coordinates of staples, find coordinates of each constituent spherocylinder
__global__ void split_staple_2_sph(int nStaples, double dGamma, double *pdX, double *pdY, 
		    double *pdPhi, double *pdR, double *pdAs, double *pdAb, double *pdCOM, 
		    double *pdSpX,double *pdSpY,double *pdSpPhi, double *pdSpR, double *pdSpA)
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

    // Coordinates of the spine
    double dDeltaY = dCOM * cos(dPhi);
    pdSpY[3*thid] = dY + dDeltaY;
    // The coordinates of the staples and spherocylinders are both stored in the shear frame,
    // which must be taken into account if gamma is nonzero
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

///////////////////////////////////////////////////////////////
// Find the Cell ID for each spherocylinder:
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
    
    // If the spherocylinders are outside the box limits, send them back inside using periodicity
    if (dY > dL) {
      dY -= dL;
      pdY[nPID] = dY;
    }
    else if (dY < 0) {
      dY += dL;
      pdY[nPID] = dY;
    }
    if (dX > dL) {
      dX -= dL;
      pdX[nPID] = dX;
    }
    else if (dX < 0) {
      dX += dL;
      pdX[nPID] = dX;
    }

    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    pnCellID[nPID] = nCellID;

    // Add 1 particle to a cell
    // atomicAdd locks the memory from other threads, completes the addition,
    //  and returns the value from before the addition, not the actual result of the addition
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    // only add particle to cell list if there is not already the maximum number in cell
    // the particle can still 'see' its neighbors that are in the cell list, but other particles in the cell won't 'see' it (so make sure maxPPC is a reasonable number)
    if (nPPC < nMaxPPC)
      pnCellList[nCellID * nMaxPPC + nPPC] = nPID;
    else
      nPPC = atomicAdd(pnPPC + nCellID, -1);

    nPID += nThreads;
  }
}

////////////////////////////////////////////////////////////////
// Here a list of possible contacts is created for each particle
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
      // First check the cell that the particle is in
      int nCellID = pnCellID[nPID];
      int nP = pnPPC[nCellID];
      for (int p = 0; p < nP; p++) {
	int nAdjPID = pnCellList[nCellID*nMaxPPC + p];
	if (nAdjPID / 3 != nPID / 3) {
	  double dSigma = dR + dA + pdR[nAdjPID] + pdA[nAdjPID] + dEpsilon;
	  double dDeltaY = dY - pdY[nAdjPID];
	  dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
	  
	  if (fabs(dDeltaY) < dSigma) {
	    double dDeltaX = dX - pdX[nAdjPID];
	    dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
	    double dDeltaRx = dDeltaX + dGamma * dDeltaY;
	    double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
	    if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma) {
	      if (nNbrs < nMaxNbrs) {
		pnNbrList[3*nStaples * nNbrs + nPID] = nAdjPID;  // This indexing makes for more coalesced reads from global memory
		nNbrs += 1;
	      }
	    }
	  }
	}
      }
      
      // Now check the eight surrounding cells
      for (int nc = 0; nc < 8; nc++) {
	int nAdjCID = pnAdjCells[8 * nCellID + nc];
	nP = pnPPC[nAdjCID];
	for (int p = 0; p < nP; p++) {
	  int nAdjPID = pnCellList[nAdjCID*nMaxPPC + p];
	  if (nAdjPID / 3 != nPID / 3) {
	    // The maximum distance at which two particles could contact
	    //  plus a little bit of moving room - dEpsilon 
	    double dSigma = dR + dA + pdA[nAdjPID] + pdR[nAdjPID] + dEpsilon;
	    double dDeltaY = dY - pdY[nAdjPID];
	    
	    // Make sure were finding the closest separation
	    dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
	    
	    if (fabs(dDeltaY) < dSigma) {
	      double dDeltaX = dX - pdX[nAdjPID];
	      dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
	      
	      // Go to unsheared coordinates
	      double dDeltaRx = dDeltaX + dGamma * dDeltaY;
	      // Also look at distance when the strain parameter is at its max (0.5)
	      double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
	      if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma) {
		if (nNbrs < nMaxNbrs) {
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


__global__ void find_contact(int nStaples, int *pnNPP, int *pnNbrList, double dL,
			     double dGamma, double *pdX, double *pdY, double *pdPhi,
			     double *pdR, double *pdA, int *nContacts)
{
  // Declare shared memory pointer, the size is determined at the kernel launch
  extern __shared__ int sCont[];
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  sCont[thid] = 0;

  while (nPID < 3*nStaples)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dPhi = pdPhi[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      
      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * 3*nStaples];
	  
	  double dDeltaX = dX - pdX[nAdjPID];
	  double dDeltaY = dY - pdY[nAdjPID];
	  double dPhiB = pdPhi[nAdjPID];
	  double dSigma = dR + pdR[nAdjPID];
	  double dB = pdA[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dGamma * dDeltaY;

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
	  if (dDSqr < dSigma*dSigma)
	    {
	      sCont[thid] += 1;
	    }
	}	
      
      nPID += nThreads;
    }
  __syncthreads();
  
  // Now we do a parallel reduction sum to find the total number of contacts
  int stride = blockDim.x / 3;
  if (thid < stride) {
    sCont[thid] += sCont[thid + 2*stride];
  }
  __syncthreads();
  while (stride > 32)
    {
      if (thid < stride)
	{
	  sCont[thid] += sCont[thid + stride];
	}
      stride /= 2;  
      __syncthreads();
    }
  if (thid < 32) //unroll end of loop
    {
      sCont[thid] += sCont[thid + 32];
      if (thid < 16)
	{
	  sCont[thid] += sCont[thid + 16];
	  if (thid < 8)
	    {
	      sCont[thid] += sCont[thid + 8];
	      if (thid < 4)
		{
		  sCont[thid] += sCont[thid + 4];
		  if (thid < 2) {
		    sCont[thid] += sCont[thid + 2];
		    if (thid == 0) {
		      sCont[0] += sCont[1];
		      int nBContacts = atomicAdd(nContacts, sCont[0]);
		    }
		  }
		}
	    }
	}
    } 
 
}


// This calculates a list of which cells are next to each other
//  and only needs to be once for convenience since the periodic boundaries make it a bit tricky
void find_adj_cells(int nCells, int nCellRows, int nCellCols, int *d_pnAdjCells)
{
  int *h_pnAdjCells = new int[8*nCells];
  for (int c = 0; c < nCells; c++)
    {
      int nRow = c / nCellCols; 
      int nCol = c % nCellCols;

      int nAdjCol1 = (nCol + 1) % nCellCols;
      int nAdjCol2 = (nCellCols + nCol - 1) % nCellCols;
      h_pnAdjCells[8 * c] = nRow * nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 1] = nRow * nCellCols + nAdjCol2;

      int nAdjRow = (nRow + 1) % nCellRows;
      h_pnAdjCells[8 * c + 2] = nAdjRow * nCellCols + nCol;
      h_pnAdjCells[8 * c + 3] = nAdjRow * nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 4] = nAdjRow * nCellCols + nAdjCol2;
      
      nAdjRow = (nCellRows + nRow - 1) % nCellRows;
      h_pnAdjCells[8 * c + 5] = nAdjRow * nCellCols + nCol;
      h_pnAdjCells[8 * c + 6] = nAdjRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 7] = nAdjRow * m_nCellCols + nAdjCol2;
    }
  cudaMemcpy(d_pnAdjCells, h_pnAdjCells, 8*m_nCells*sizeof(int), cudaMemcpyHostToDevice);
  delete[] h_pnAdjCells;
}



int main(int argc, char *argv[]) 
{
  int nStaples = 1024;
  double dGamma = 0.0;
  // The grid and block size specify how to launch kernel function on the gpu
  int nStBlockSize = 128;
  int nSpBlockSize = 192;
  int nStGridSize = 8;
  int nSpGridSize = 16;
  // These will point to the list of coordinates, arrays designated h_* will be allocated 
  //  on the cpu (host), and those designated d_* will be allocated on the gpu (device)
  double *h_pdX, *h_pdY, *h_pdPhi, *d_pdX, *d_pdY, *d_pdPhi;
  // cudaHostAlloc allocates on the cpu but allows for different types of memory transfers, 
  //  such as asynchronous transfers, which can be faster for frequently transfered data
  cudaHostAlloc((void**) &h_pdX, nStaples*sizeof(double), 0);  
  cudaHostAlloc((void**) &h_pdY, nStaples*sizeof(double), 0); 
  cudaHostAlloc((void**) &h_pdPhi, nStaples*sizeof(double), 0);
  // cudaMalloc allocates memory on the gpu
  cudaMalloc((void**) &d_pdX, nStaples*sizeof(double), 0);  
  cudaMalloc((void**) &d_pdY, nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdPhi, nStaples*sizeof(double), 0);
  
  // The radius and spine and barb lengths
  double *h_pdR = new double[nStaples];  
  double *h_pdAs = new double[nStaples];
  double *h_pdAb = new double[nStaples]; 
  double *d_pdR, *d_pdAs, *d_pdAb;
  cudaMalloc((void**) &d_pdR, nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdAs, nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdAb, nStaples*sizeof(double), 0);
  // Several rotational constants such as the center of mass
  double *d_pdCOM, *d_pdMOI, *d_pdSinCoeff, *d_pdCosCoeff;
  cudaMalloc((void**) &d_pdCOM, nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdMOI, nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdSinCoeff, nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdCosCoeff, nStaples*sizeof(double), 0);

  // Get some random positions
  double dPacking = 0.4;
  double dR = 0.5;
  double dA = 2.0;
  const double pi = 3.141592653589793;
  double dL = sqrt(nStaples*(12.0 + 3*pi*0.25) / dPacking);
  srand(time(0));
  for (int s = 0; s < nStaples; s++) {
    h_pdR[s] = dR;
    h_pdAs[s] = dA;
    h_pdAb[s] = dA;
    h_pdX[s] = dL * static_cast<double>(rand()) / std::RAND_MAX;
    h_pdY[s] = dL * static_cast<double>(rand()) / std::RAND_MAX;
    h_pdPhi[s] = 2*pi * static_cast<double>(rand()) / std::RAND_MAX;
  }
  // Copy the positions from the cpu to the gpu
  cudaMemcpy(d_pdR, h_pdR, nStaples*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pdAs, h_pdAs, nStaples*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pdAb, h_pdAb, nStaples*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdX, h_pdX, nStaples*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, nStaples*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, nStaples*sizeof(double), cudaMemCpyHostToDevice);

  // find the center of mass, moment of inertia, etc for each staple
  calc_rotational_consts <<<nStGridSize, nStBlockSize>>> 
    (nStaples, d_pdR, d_pdAs, d_dAb, d_pdCOM, d_pdMOI, d_pdSinCoeff, d_pdCosCoeff);
  
  // These arrays are the coordinates of the constituent spherocylinders
  double *d_pdSPX, *d_pdSpY, *d_pdSpPhi, *d_pdSpR, d_pdSpA;
  cudaMalloc((void**) &d_pdSpX, 3*nStaples*sizeof(double), 0);  
  cudaMalloc((void**) &d_pdSpY, 3*nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdSpPhi, 3*nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdSpR, 3*nStaples*sizeof(double), 0);
  cudaMalloc((void**) &d_pdSpA, 3*nStaples*sizeof(double), 0);
 
  // Block the cpu from executing code until all threads involved in kernels 
  //  or asynchronous memory transfers have finished
  cudaThreadSynchronize();  
  
  split_staples_2_sph <<<nStGridSize, nSpGridSize>>> 
    (nStaples, dGamma, d_pdX, d_pdY, d_pdPhi,d_pdR, d_pdAs, d_pdAb, d_pdCOM, 
     d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA);
  
  // Here we calculate the division of the box into cells
  double dEpsilon = 0.1;  // Cell padding so we don't need to recalculate neighbors all of the time
  double dWMin = 2.24 * (dR + dA) + dEpsilon;  // Width must take into account shortest distance across the cell for the maximum value of gamma (0.5)
  double dHMin = 2 * (dR + dA) + dEpsilon;

  int nCellRows = max(static_cast<int>(m_dL / dHMin), 1);
  int nCellCols = max(static_cast<int>(m_dL / dWMin), 1);
  int nCells = m_nCellRows * m_nCellCols;
  double dCellW = dL / nCellRows;
  double dCellH = dL / nCellCols;

  // Allocate list of particles per cell, lists of particle indices for each cell, a
  int *d_pnPPC, *d_pnCellList, *d_pnCellID;
  int nMaxPPC = 20;
  cudaMalloc((void**) &d_pnCellID, nStaples*sizeof(int));
  cudaMalloc((void**) &d_pnPPC, nCells*sizeof(int));
  cudaMemset(d_pnPPC, 0, nCells*sizeof(int));  // Initialize array to zero
  cudaMalloc((void**) &d_pnCellList, nCells*nMaxPPC*sizeof(int));

  cudaThreadSynchronize();

  find_cells <<<nSpGridSize, nSpGridSize>>> (nStaples, nMaxPPC, dCellW, dCellH, nCellCols, dL, 
					     d_pdSpX, d_pdSpY, d_pnCellID, d_pnPPC, d_pnCellList);

  int *d_pnNPP, *d_pnNeighborList, *d_pnAdjCells;
  int nMaxNPP = 20;
  cudaMalloc((void**) &d_pnNPP, nStaples*sizeof(int));
  cudaMalloc((void**) &d_pnNeighborList, nStaples*nMaxNPP*sizeof(int));
  cudaMalloc((void**) &d_pnAdjCells, 8*nCells*sizeof(int));
  
  find_adj_cells(nCells, nCellRows, nCellCols, d_pnAdjCells);

  cudaThreadSynchronize();

  find_nbrs <<<nSpGridSize, nSpGridSize>>> 
    (nStaples, nMaxPPC, d_pnCellID, d_pnPPC, d_pnCellList, d_pnAdjCells, nMaxNPP, d_pnNPP, 
     d_pnNbrList, d_pdSpX, d_pdSpY, d_pdSpR, d_pdSpA, dEpsilon, dL, dGamma);

  cudaThreadSynchronize();


  // Free all of the allocated memory
  cudaFreeHost(h_pdX); cudaFreeHost(h_pdY); cudaFreeHost(h_pdPhi);
  cudaFree(d_pdX); cudaFree(d_pdY); cudaFree(d_pdPhi);
  delete[] h_pdR; delete[] h_pdAs; delete[] h_pdAb;
  cudaFree(d_pdR); cudaFree(d_pdAs); cudaFree(d_pdAb);
  cudaFree(d_pdCOM); cudaFree(d_pdMOI); cudaFree(d_pdSinCoeff); cudaFree(d_pdCosCoeff);
  cudaFree(d_pdSpX); cudaFree(d_pdSpY); cudaFree(d_pdSpPhi); cudaFree(d_pdSpR); cudaFree(d_pdSpA);
  cudaFree(d_pnPPC); cudaFree(d_pnCellList); cudaFree(d_pnAdjCells); cudaFree(d_pnCellID);
  cudaFree(d_pnNPP); cudaFree(d_pnNeighborList);

  return 0;
}
