// -*- c++ -*-
/*
* calculate_stress_energy.cu
*
*
*/

#include <cuda.h>
#include "staple_box.h"
#include "cudaErr.h"
#include <math.h>
#include <sm_20_atomic_functions.h>
#include <stdio.h>


using namespace std;


////////////////////////////////////////////////////////
// Calculates energy, stress tensor, forces:
//  Returns returns energy, pxx, pyy and pxy to pfSE
//  Returns 
//
// The neighbor list pnNbrList has a list of possible contacts
//  for each particle and is found in find_neighbors.cu
//
/////////////////////////////////////////////////////////
template<Potential ePot>
__global__ void calc_se(int nStaples, int *pnNPP, int *pnNbrList, double dL,
			double dGamma, double *pdX, double *pdY, double *pdPhi,
			double *pdR, double *pdA, double *pdXcom, double *pdYcom, 
			double *pdFx, double *pdFy, double *pdFt, int *pnContacts, float *pfSE)
{
  // Declare shared memory pointer, the size is determined at the kernel launch
  extern __shared__ double sData[];
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  int offset = 4 * blockDim.x / 3 + 8; // +8 helps to avoid bank conflicts
  for (int i = thid; i < 4*offset + 4*blockDim.x; i += blockDim.x)
    sData[i] = 0.0;
  __syncthreads();  // synchronizes every thread in the block before going on

  while (nPID < 3*nStaples)
    {
      double dFx = 0.0;
      double dFy = 0.0;
      double dFt = 0.0;
      
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
	      sData[4*offset + 3*blockDim.x + thid] += 1.0;
	      double dDij = sqrt(dDSqr);
	      double dDVij;
	      double dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDx * dDVij / dDij;
	      double dPfy = dDy * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      double dDeltaXcom = dX - pdXcom[nPID / 3];
	      double dDeltaYcom = dY  - pdYcom[nPID / 3];
	      dDeltaXcom += dL * ((dDeltaXcom < -0.5*dL) - (dDeltaXcom > 0.5*dL));
	      dDeltaYcom += dL * ((dDeltaYcom < -0.5*dL) - (dDeltaYcom > 0.5*dL));
	      dDeltaXcom += dGamma * dDeltaYcom + s*nxA;
	      dDeltaYcom += s*nyA;
	      dFt += dDeltaXcom * dPfy - dDeltaYcom * dPfx;
	      if (nAdjPID > nPID)
		{
		  sData[thid] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * nStaples);
		  sData[thid + offset] += dPfx * dDx / (dL * dL);
		  sData[thid + 2*offset] += dPfy * dDy / (dL * dL);
		  sData[thid + 3*offset] += dPfx * dDy / (dL * dL);
		} 
	    }
	}
      sData[4*offset + thid] = dFx;
      sData[4*offset + blockDim.x + thid] = dFy;
      sData[4*offset + 2*blockDim.x + thid] = dFt;
      __syncthreads();
      int nF = thid % 3;
      int nSt = thid / 3;
      int b = 4*offset + nF*blockDim.x + 3*nSt;
      sData[b] += sData[b + 1] + sData[b + 2];
      switch (nF) 
	{
	case 0:
	  pdFx[nPID / 3] = sData[b];
	  pnContacts[nPID / 3] = int(sData[4*offset + 3*blockDim.x + thid]
				     + sData[4*offset + 3*blockDim.x + thid + 1]
				     + sData[4*offset + 3*blockDim.x + thid + 2]);
	  break;
	case 1:
	  pdFy[nPID / 3] = sData[b];
	  break;
	case 2:
	  pdFt[nPID / 3] = sData[b];
	}	
      
      nPID += nThreads;
    }
  __syncthreads();
  
  // Now we do a parallel reduction sum to find the total number of contacts
  int stride = 2*blockDim.x / 3;
  int base = thid;
  if (thid < stride) {
    while (base < 4*offset) {
      sData[base] += sData[base + stride];
      base += offset;
    }
  }
  stride /= 2; // stride is 1/4 block size, all threads perform 1 add
  __syncthreads();
  base = thid % stride + offset * (thid / stride);
  if (thid < 2*stride) {
    sData[base] += sData[base + stride];
    base += 2*offset;
    sData[base] += sData[base + stride];
  }
  stride /= 2;
  __syncthreads();
  while (stride > 8)
    {
      if (thid < 4 * stride)
	{
	  base = thid % stride + offset * (thid / stride);
	  sData[base] += sData[base + stride];
	}
      stride /= 2;  
      __syncthreads();
    }
  if (thid < 32) //unroll end of loop
    {
      base = thid % 8 + offset * (thid / 8);
      sData[base] += sData[base + 8];
      if (thid < 16)
	{
	  base = thid % 4 + offset * (thid / 4);
	  sData[base] += sData[base + 4];
	  if (thid < 8)
	    {
	      base = thid % 2 + offset * (thid / 2);
	      sData[base] += sData[base + 2];
	      if (thid < 4)
		{
		  sData[thid * offset] += sData[thid * offset + 1];
		  float tot = atomicAdd(pfSE+thid, (float)sData[thid*offset]);	    
		}
	    }
	}
    }  
}

__global__ void avg_cont_velo(int nStaples, double *pdFt, int *pnContacts, double *pdMOI, float *dAvgW, int *nTotC) {
  extern __shared__ double sMem[];
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nStride = blockDim.x / 2;
  int offset = nStride;
  if (nPID < nStaples) {
    if (thid < nStride) {
      sMem[thid] = pdFt[nPID] / pdMOI[nPID] + pdFt[nPID + nStride] / pdMOI[nPID + nStride];
    }
    else {
      sMem[thid] = double(pnContacts[nPID] + pnContacts[nPID - nStride]);
    }
    __syncthreads();

    nStride /= 2;
    while (nStride > 16) {
      int nBase = thid % nStride + offset * (thid / nStride);
      if (thid < 2 * nStride)
	sMem[nBase] += sMem[nBase + nStride];
      nStride /= 2;
      __syncthreads();
    }
	  
    if (thid < 32) {
      int nBase = thid % 16 + offset * (thid / 16);
      sMem[nBase] += sMem[nBase + 16];
      if (thid < 16) {
	nBase = thid % 8 + offset * (thid / 8);
	sMem[nBase] += sMem[nBase + 8];
	if (thid < 8) {
	  nBase = thid % 4 + offset * (thid / 4);
	  sMem[nBase] += sMem[nBase + 4];
	  if (thid < 4) {
	    nBase = thid % 2 + offset * (thid / 2);
	    sMem[nBase] += sMem[nBase + 2];
	    if (thid < 2) {
	      nBase = offset * thid;
	      sMem[nBase] += sMem[nBase + 1];
	      if (thid == 0) {
	        float ftot = atomicAdd(dAvgW, float(sMem[0] / nStaples));
		int ntot = atomicAdd(nTotC, int(sMem[offset]+0.5));
	      }
	    }
	  }
	}
      }
    }

  }
}


void Staple_Box::calculate_stress_energy()
{
  cudaMemset((void*) d_pfSE, 0, 4*sizeof(float));
  cudaMemset((void *) d_pfAvgAngVelo, 0, sizeof(float));
  cudaMemset((void *) d_pnTotContacts, 0, sizeof(int));
  
  //dim3 grid(m_nGridSize);
  //dim3 block(m_nBlockSize);
  //size_t smem = m_nSM_CalcSE;
  //printf("Configuration: %d x %d x %d\n", m_nGridSize, m_nBlockSize, m_nSM_CalcSE);

  switch (m_ePotential)
    {
    case HARMONIC:
      calc_se <HARMONIC> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, d_pdSpX, d_pdSpY, d_pdSpPhi, 
	 d_pdSpR, d_pdSpA, d_pdX, d_pdY, d_pdFx, d_pdFy, d_pdFt, d_pnContacts, d_pfSE);
      break;
    case HERTZIAN:
      calc_se <HERTZIAN> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, d_pdSpX, d_pdSpY, d_pdSpPhi, 
	 d_pdSpR, d_pdSpA, d_pdX, d_pdY, d_pdFx, d_pdFy, d_pdFt, d_pnContacts, d_pfSE);
    }
  cudaThreadSynchronize();
  checkCudaError("Calculating stresses and energy");

  avg_cont_velo <<<m_nGridSize, m_nBlockSize, (16+m_nBlockSize)*sizeof(double)>>>
	(m_nStaples, d_pdFt, d_pnContacts, d_pdMOI, d_pfAvgAngVelo, d_pnTotContacts);
  cudaThreadSynchronize();
  checkCudaError("Summing contacts and angular velocity");

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

bool Staple_Box::check_for_contacts()
{
  int *pnContacts;
  cudaMalloc((void**) &pnContacts, sizeof(int));
  cudaMemset((void*) pnContacts, 0, sizeof(int));
  
  int nSMSize = m_nSpBlockSize * sizeof(int);
  find_contact <<<m_nSpGridSize, m_nSpBlockSize, nSMSize>>>
    (m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, d_pdSpX, d_pdSpY, 
     d_pdSpPhi, d_pdSpR, d_pdSpA, pnContacts);
  cudaThreadSynchronize();

  int nContacts;
  cudaMemcpy(&nContacts, pnContacts, sizeof(int), cudaMemcpyHostToDevice);
  cudaFree(pnContacts);

  return (bool)nContacts;
}











////////////////////////////////////////////////////////////////////////////////////
#if GOLD_FUNCS == 1

void calc_se_gold(Potential ePot, int gridDim, int blockDim, int sMemSize, 
		  int nStaples, int *pnNPP, int *pnNbrList, double dL, double dGamma, 
		  double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdA, 
		  double *pdXcom, double *pdYcom, double *pdFx, double *pdFy, double *pdFt, float *pfSE)
{
  for (int blockIdx = 0; blockIdx < gridDim; blockIdx++) {
    printf("Entering loop, block %d\n", blockIdx);
    double *sData = (double*)malloc(sMemSize);
    int offset = 4 * blockDim / 3 + 8;
    int nThreads = blockDim * gridDim;
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      printf("Entering loop, thread %d\n", threadIdx);
      // Declare shared memory pointer, the size is determined at the kernel launch      
      int thid = threadIdx;
      int nPID = thid + blockIdx * blockDim;
      for (int i = thid; i < 4*offset + 3*blockDim; i += blockDim)
	sData[i] = 0.0;
    }
    printf("Synchronizing threads in block %d (shared memory set)\n", blockIdx);
    
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      printf("Entering loop, thread %d\n", threadIdx);
      // Declare shared memory pointer, the size is determined at the kernel launch      
      int thid = threadIdx;
      int nPID = thid + blockIdx * blockDim;
      
      double dFx = 0.0;
      double dFy = 0.0;
      double dFt = 0.0;
      
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dPhi = pdPhi[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      
      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++) {
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
	if (dDSqr < dSigma*dSigma) {
	  double dDij = sqrt(dDSqr);
	  double dDVij;
	  double dAlpha;
	  if (ePot == HARMONIC) {
	    dDVij = (1.0 - dDij / dSigma) / dSigma;
	    dAlpha = 2.0;
	  }
	  else if (ePot == HERTZIAN) {
	    dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
	    dAlpha = 2.5;
	  }
	  double dPfx = dDx * dDVij / dDij;
	  double dPfy = dDy * dDVij / dDij;
	  dFx += dPfx;
	  dFy += dPfy;
	  double dDeltaXcom = dX - pdXcom[nPID / 3];
	  double dDeltaYcom = dY  - pdYcom[nPID / 3];
	  dDeltaXcom += dL * ((dDeltaXcom < -0.5*dL) - (dDeltaXcom > 0.5*dL));
	  dDeltaYcom += dL * ((dDeltaYcom < -0.5*dL) - (dDeltaYcom > 0.5*dL));
	  dDeltaXcom += dGamma * dDeltaYcom + s*nxA;
	  dDeltaYcom += s*nyA;
	  dFt += dDeltaXcom * dPfy - dDeltaYcom * dPfx;
	  if (nAdjPID > nPID) {
	    sData[thid] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * nStaples);
	    sData[thid + offset] += dPfx * dDx / (dL * dL);
	    sData[thid + 2*offset] += dPfy * dDy / (dL * dL);
	    sData[thid + 3*offset] += dPfx * dDy / (dL * dL);
	  } 
	}
      }
      sData[4*offset + thid] = dFx;
      sData[4*offset + blockDim + thid] = dFy;
      sData[4*offset + 2*blockDim + thid] = dFt;
    }
    printf("Threads in block %d synchronized (contact forces calculated)\n", blockIdx);
    
    for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
      printf("Entering loop, thread %d\n", threadIdx);
      // Declare shared memory pointer, the size is determined at the kernel launch      
      int thid = threadIdx;
      int nPID = thid + blockIdx * blockDim;
      
      int nF = thid % 3;
      int nSt = thid / 3;
      int b = 4*offset + nF*blockDim + 3*nSt;
      sData[b] += sData[b + 1] + sData[b + 2];
      switch (nF) 
	{
	case 0:
	  pdFx[nPID / 3] = sData[b];
	  break;
	case 1:
	  pdFy[nPID / 3] = sData[b];
	  break;
	case 2:
	  pdFt[nPID / 3] = sData[b];
	}	 
    }

    // Now we do a parallel reduction sum to find the total number of contacts
    for (int t = 1; t < blockDim; t++) { 
      sData[0] += sData[t];
      sData[offset] += sData[offset + t];
      sData[2*offset] += sData[2*offset + t];
      sData[3*offset] += sData[3*offset + t];
    }    
    
    for (int s = 0; s < 4; s++)
      pfSE[s] += (float)sData[s*offset];
   
    free(sData);
  }
}

void Staple_Box::calculate_stress_energy_gold()
{
  for (int s = 0; s < 4; s++)
    g_pfSE[s] = 0;

  printf("Calculating stresses and energy");
  cudaMemcpy(g_pnNPP, d_pnNPP, sizeof(int)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pnNbrList, d_pnNbrList, sizeof(int)*m_nStaples*m_nMaxNbrs, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdX, d_pdX, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdY, d_pdY, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdPhi, d_pdPhi, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdR, d_pdR, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdAspn, d_pdAspn, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdAbrb, d_pdAbrb, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdSpX, d_pdSpX, 3*sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdSpY, d_pdSpY, 3*sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdSpPhi, d_pdSpPhi, 3*sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdSpR, d_pdSpR, 3*sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdSpA, d_pdSpA, 3*sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);

  switch (m_ePotential)
    {
    case HARMONIC:
      calc_se_gold (HARMONIC, m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE,
		    m_nStaples, g_pnNPP, g_pnNbrList, m_dL, m_dGamma, g_pdSpX, g_pdSpY, 
		    g_pdSpPhi, g_pdSpR, g_pdSpA, g_pdX, g_pdY, g_pdFx, g_pdFy, g_pdFt, g_pfSE);
      break;
    case HERTZIAN:
      calc_se_gold (HERTZIAN, m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE, 
		    m_nStaples, g_pnNPP, g_pnNbrList, m_dL, m_dGamma, g_pdSpX, g_pdSpY, 
		    g_pdSpPhi, g_pdSpR, g_pdSpA, g_pdX, g_pdY, g_pdFx, g_pdFy, g_pdFt, g_pfSE);
    }
  
}

void Staple_Box::compare_calculate_stress_energy()
{
  calculate_stress_energy();
  cudaMemcpy(h_pdFx, d_pdFx, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdFy, d_pdFy, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdFt, d_pdFt, sizeof(double)*m_nStaples, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pfSE, d_pfSE, sizeof(float)*4, cudaMemcpyDeviceToHost);

  calculate_stress_energy_gold();

  double dTol = 0.00001;
  for (int s = 0; s < 4; s++) {
    if (fabs(h_pfSE[s]) < fabs(g_pfSE[s]) * (1. - dTol) || fabs(h_pfSE[s]) > fabs(g_pfSE[s]) * (1. + dTol) ) {
      printf("Difference found in stress %d:\n", s);
      printf("GPU: %.7g, gold: %.7g\n", h_pfSE[s], g_pfSE[s]);
    }
  }
  for (int p = 0; p < m_nStaples; p++) {
    if (fabs(h_pdFx[p]) < fabs(g_pdFx[p]) * (1. - dTol) || fabs(h_pdFx[p]) > fabs(g_pdFx[p]) * (1. + dTol) ||
	fabs(h_pdFy[p]) < fabs(g_pdFy[p]) * (1. - dTol) || fabs(h_pdFy[p]) > fabs(g_pdFy[p]) * (1. + dTol) ||
	fabs(h_pdFt[p]) < fabs(g_pdFt[p]) * (1. - dTol) || fabs(h_pdFt[p]) > fabs(g_pdFt[p]) * (1. + dTol) ) {
      printf("Difference found in calculated force for particle %d:\n", p);
      printf("GPU: %.9g, %.9g, %.9g\n", h_pdFx[p], h_pdFy[p], h_pdFt[p]);
      printf("Gld: %.9g, %.9g, %.9g\n", g_pdFx[p], g_pdFy[p], g_pdFt[p]);
    }
  }
  
}

#endif
