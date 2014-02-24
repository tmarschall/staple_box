
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
#include <string.h>
#include <stdlib.h>
#include <string>

using namespace std;

const double D_PI = 3.14159265358979;

/*
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
    double dB = pdAb[nPID];
    double dC = pdCOM[nPID];

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
*/

///////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////
template<Potential ePot, int bCalcStress>
__global__ void euler_est(int nStaples, int *pnNPP, int *pnNbrList, double dL, double dGamma, 
			  double dStrain, double dStep, double *pdX, double *pdY, double *pdPhi, 
			  double *pdR, double *pdA, double *pdCOM, double *pdMOI, double *pdSinCoef, 
			  double *pdCosCoef, double *pdXcom, double *pdYcom, double *pdPhicom, 
			  double *pdFx, double *pdFy, double *pdFt, int *pnContacts, float *pfSE, 
			  double *pdTempX, double *pdTempY, double *pdTempPhi)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  int offset = 4 * blockDim.x / 3 + 8; // +8 should help to avoid a few bank conflicts
  sData[thid] = 0.0;
  sData[blockDim.x+thid] = 0.0;
  sData[2*blockDim.x+thid] = 0.0;
  sData[3*blockDim.x+thid] = 0.0;
  if (bCalcStress) {
    for (int i = thid; i < 4*offset; i += blockDim.x)
      sData[4*blockDim.x + i] = 0.0;
  }
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
	  dDeltaX += dL * (double(dDeltaX < -0.5*dL) - double(dDeltaX > 0.5*dL));
	  dDeltaY += dL * (double(dDeltaY < -0.5*dL) - double(dDeltaY > 0.5*dL));
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
	      sData[3*blockDim.x + thid] += 1.0;
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
	      double dDeltaYcom = dY - pdYcom[nPID / 3];
	      dDeltaXcom += dL * (double(dDeltaXcom < -0.5*dL) - double(dDeltaXcom > 0.5*dL));
	      dDeltaYcom += dL * (double(dDeltaYcom < -0.5*dL) - double(dDeltaYcom > 0.5*dL));
	      dDeltaXcom += dGamma * dDeltaYcom + s*nxA;
	      dDeltaYcom += s*nyA;
	      dFt += dDeltaXcom * dPfy - dDeltaYcom * dPfx;
	      if (bCalcStress)
		{
		  if (nAdjPID > nPID)
		    {
		      sData[4*blockDim.x + thid] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * nStaples);
		      sData[4*blockDim.x + thid + offset] += dPfx * dDx / (dL * dL);
		      sData[4*blockDim.x + thid + 2*offset] += dPfy * dDy / (dL * dL);
		      sData[4*blockDim.x + thid + 3*offset] += dPfx * dDy / (dL * dL);
		    } 
		}
	    }
	}
      sData[thid] = dFx;
      sData[blockDim.x + thid] = dFy;
      sData[2*blockDim.x + thid] = dFt;
      __syncthreads();
      // sum the spherocylinder forces for each staple
      int nF = thid % 3;
      int nSt = thid / 3;
      int b = nF*blockDim.x + 3*nSt;
      sData[b] += sData[b + 1] + sData[b + 2];
      switch (nF) 
	{
	case 0:
	  pdFx[nPID / 3] = sData[b];
	  pnContacts[nPID / 3] = int(sData[3*blockDim.x + thid] 
				     + sData[3*blockDim.x + thid + 1] 
				     + sData[3*blockDim.x + thid + 2] + 0.5);
	  break;
	case 1:
	  pdFy[nPID / 3] = sData[b];
	  break;
	case 2:
	  pdFt[nPID / 3] = sData[b];
	}
      __syncthreads();

      if (nF == 0) {
	dFx = sData[thid];
	dFy = sData[blockDim.x + thid];
	dFt = sData[2*blockDim.x + thid] / pdMOI[nPID/3];
	
	pdTempX[nPID / 3] = pdXcom[nPID / 3] + dStep * (dFx - dGamma * dFy);
	pdTempY[nPID / 3] = pdYcom[nPID / 3] + dStep * dFy;
	double dPhiC = pdPhicom[nPID / 3];
	double dCosPhi = cos(dPhiC);
	double dSinPhi = sin(dPhiC);
	double dFtDis = pdCosCoef[nPID/3] * dCosPhi * dCosPhi 
	  + pdSinCoef[nPID/3] * dSinPhi * dSinPhi;
	pdTempPhi[nPID / 3] = dPhiC + dStep * (dFt - dStrain * dFtDis);
      }
      
      nPID += nThreads;
    }

  if (bCalcStress) {
    __syncthreads();
    
    // Now we do a parallel reduction sum to find the total number of contacts
    int stride = 2*blockDim.x / 3;
    int base = 4*blockDim.x + thid;
    if (thid < stride) {
      while (base < 4*blockDim.x + 4*offset) {
	sData[base] += sData[base + stride];
	base += offset;
      }
    }
    stride /= 2; // stride is 1/4 block size, all threads perform 1 add
    __syncthreads();

    base = 4*blockDim.x + thid % stride + offset * (thid / stride);
    if (thid < 2*stride) {
      sData[base] += sData[base + stride];
      base += 2*offset;
      sData[base] += sData[base + stride];
    }
    stride /= 2;
    __syncthreads();

    while (stride > 8) {
      if (thid < 4 * stride) {
	base = 4*blockDim.x + thid % stride + offset * (thid / stride);
	sData[base] += sData[base + stride];
      }
      stride /= 2;  
      __syncthreads();
    }

    if (thid < 32) { //unroll end of loop
      base = 4*blockDim.x + thid % 8 + offset * (thid / 8);
      sData[base] += sData[base + 8];
      if (thid < 16) {
	base = 4*blockDim.x + thid % 4 + offset * (thid / 4);
	sData[base] += sData[base + 4];
	if (thid < 8) {
	  base = 4*blockDim.x + thid % 2 + offset * (thid / 2);
	  sData[base] += sData[base + 2];
	  if (thid < 4) {
	    sData[4*blockDim.x + thid * offset] += sData[4*blockDim.x + thid * offset + 1];
	    float tot = atomicAdd(pfSE+thid, (float)sData[4*blockDim.x + thid*offset]);	    
	  }
	}
      }
    }  
    
  }
}

///////////////////////////////////////////////////////////////////
//
//
/////////////////////////////////////////////////////////////////
template<Potential ePot>
__global__ void heun_corr(int nStaples, int *pnNPP, int *pnNbrList, double dL, double dGamma, 
			  double dStrain, double dStep, double *pdX, double *pdY, double *pdPhi, 
			  double *pdR, double *pdA, double *pdCOM, double *pdMOI, double *pdSinCoef, 
			  double *pdCosCoef, double *pdXcom, double *pdYcom, double *pdPhicom, 
			  double *pdFx, double *pdFy, double *pdFt, double *pdTempX, double *pdTempY)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  sData[thid] = 0.0;
  sData[blockDim.x+thid] = 0.0;
  sData[2*blockDim.x+thid] = 0.0;
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
      double dNewGamma = dGamma + dStep * dStrain;

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
	  dDeltaX += dL * (double(dDeltaX < -0.5*dL) - double(dDeltaX > 0.5*dL));
	  dDeltaY += dL * (double(dDeltaY < -0.5*dL) - double(dDeltaY > 0.5*dL));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dNewGamma * dDeltaY;
	  
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
	      double dDij = sqrt(dDSqr);
	      double dDVij;
	      //double dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDij / dSigma) / dSigma;
		  //dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		  //dAlpha = 2.5;
		}
	      double dPfx = dDx * dDVij / dDij;
	      double dPfy = dDy * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      
	      double dDeltaXcom = dX - pdTempX[nPID / 3];
	      double dDeltaYcom = dY  - pdTempY[nPID / 3];
	      dDeltaXcom += dL * (double(dDeltaXcom < -0.5*dL) - double(dDeltaXcom > 0.5*dL));
	      dDeltaYcom += dL * (double(dDeltaYcom < -0.5*dL) - double(dDeltaYcom > 0.5*dL));
	      dDeltaXcom += dNewGamma * dDeltaYcom + s*nxA;
	      dDeltaYcom += s*nyA;
	      dFt += dDeltaXcom * dPfy - dDeltaYcom * dPfx;
	    }
	}
      sData[thid] = dFx;
      sData[blockDim.x + thid] = dFy;
      sData[2*blockDim.x + thid] = dFt;
      __syncthreads();
      int nF = thid % 3;
      int nSt = thid / 3;
      int b = nF*blockDim.x + 3*nSt;
      sData[b] += sData[b + 1] + sData[b + 2];
      __syncthreads();

      if (nF == 0) {
	double dMOI = pdMOI[nPID/3];
	dFy = sData[blockDim.x + thid];
	dFx = sData[thid] - dNewGamma * dFy;
	double dSinPhi = sin(dPhi);
	double dCosPhi = cos(dPhi);
	double dFtDis = pdCosCoef[nPID/3] * dCosPhi * dCosPhi 
	  + pdSinCoef[nPID/3] * dSinPhi * dSinPhi;
	dFt = sData[2*blockDim.x + thid] / dMOI - dStrain * dFtDis;
	
	double dFy0 = pdFy[nPID / 3];
	double dFx0 = pdFx[nPID / 3] - dGamma * dFy0;
	double dSinPhi0 = sin(pdPhicom[nPID / 3]);
	double dCosPhi0 = cos(pdPhicom[nPID / 3]);
	double dFtDis0 = pdCosCoef[nPID/3] * dCosPhi0 * dCosPhi0 
	  + pdSinCoef[nPID/3] * dSinPhi0 * dSinPhi0;
	double dFt0 = pdFt[nPID / 3] / dMOI - dStrain * dFtDis0;
	
	pdXcom[nPID / 3] += 0.5 * dStep * (dFx0 + dFx);
	pdYcom[nPID / 3] += 0.5 * dStep * (dFy0 + dFy);
	pdPhicom[nPID / 3] += 0.5 * dStep * (dFt0 + dFt);
      }
      __syncthreads();
      
      nPID += nThreads;
    }
}

__global__ void split_staple_2_sph(int nStaples, double dGamma, double *pdX, 
		   double *pdY, double *pdPhi, double *pdR, double *pdAs, 
                   double *pdAb, double *pdCOM, double *pdSpX, double *pdSpY, 
		   double *pdSpPhi)
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

    // Coordinates of barbs
    dDeltaY = (dAs + 2.*dR) * sin(dPhi)
      - (dAb - dCOM) * cos(dPhi);
    pdSpY[3*thid + 1] = dY + dDeltaY;
    pdSpX[3*thid + 1] = dX + (dAs + 2.*dR) * cos(dPhi) 
      + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid + 1] = dPhi + 0.5 * D_PI;

    dDeltaY = -(dAs + 2.*dR) * sin(dPhi)
      - (dAb - dCOM) * cos(dPhi);
    pdSpY[3*thid + 2] = dY + dDeltaY;
    pdSpX[3*thid + 2] = dX - (dAs + 2.*dR) * cos(dPhi) 
      + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid + 2] = dPhi - 0.5 * D_PI;

    thid += blockDim.x * gridDim.x;
  }
}

__global__ void split_staple_2_sph(int nStaples, double dGamma, double *pdX, 
		   double *pdY, double *pdPhi, double *pdR, double *pdAs, 
                   double *pdAb, double *pdCOM, double *pdSpX,double *pdSpY, 
		   double *pdSpPhi, double *pdXMoved, double *pdYMoved, 
		   int *bNewNbrs, double dEpsilon)
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
    double dSpXold = pdSpX[3*thid];
    double dSpYold = pdSpY[3*thid];
    double dDeltaY = dCOM * cos(dPhi);
    pdSpY[3*thid] = dY + dDeltaY;
    pdSpX[3*thid] = dX - dCOM * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid] = dPhi;
    pdXMoved[3*thid] += pdSpX[3*thid] - dSpXold;
    pdYMoved[3*thid] += pdSpY[3*thid] - dSpYold;

    // Coordinates of barbs
    dSpXold = pdSpX[3*thid + 1];
    dSpYold = pdSpY[3*thid + 1];
    dDeltaY = (dAs + 2.*dR) * sin(dPhi)
      - (dAb - dCOM) * cos(dPhi);
    pdSpY[3*thid + 1] = dY + dDeltaY;
    pdSpX[3*thid + 1] = dX + (dAs + 2.*dR) * cos(dPhi) 
      + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid + 1] = dPhi + 0.5 * D_PI;
    pdXMoved[3*thid+1] += pdSpX[3*thid+1] - dSpXold;
    pdYMoved[3*thid+1] += pdSpY[3*thid+1] - dSpYold;

    dSpXold = pdSpX[3*thid + 2];
    dSpYold = pdSpY[3*thid + 2];  
    dDeltaY = -(dAs + 2.*dR) * sin(dPhi)
      - (dAb - dCOM) * cos(dPhi);
    pdSpY[3*thid + 2] = dY + dDeltaY;
    pdSpX[3*thid + 2] = dX - (dAs + 2.*dR) * cos(dPhi) 
      + (dAb - dCOM) * sin(dPhi) - dGamma * dDeltaY;
    pdSpPhi[3*thid + 2] = dPhi - 0.5 * D_PI;
    pdXMoved[3*thid + 2] += pdSpX[3*thid + 2] - dSpXold;
    pdYMoved[3*thid + 2] += pdSpY[3*thid + 2] - dSpYold;
    
    if (fabs(pdXMoved[3*thid]) > 0.5*dEpsilon || fabs(pdYMoved[3*thid]) > 0.5*dEpsilon)
	*bNewNbrs = 1;
    else if (fabs(pdXMoved[3*thid+1]) > 0.5*dEpsilon || fabs(pdYMoved[3*thid+1]) > 0.5*dEpsilon)
	*bNewNbrs = 1;
    else if (fabs(pdXMoved[3*thid+2]) > 0.5*dEpsilon || fabs(pdYMoved[3*thid+2]) > 0.5*dEpsilon)
	*bNewNbrs = 1;
    
    thid += blockDim.x * gridDim.x;
  }
}

__global__ void avg_contact_velo(int nStaples, double *pdFt, int *pnContacts, double *pdMOI, float *dAvgW, int *nTotC) {
  extern __shared__ double sMem[];
  int thid = threadIdx.x;
  int nPID = thid / 2 + blockIdx.x * blockDim.x;
  int nStride = blockDim.x / 2;
  if (nPID < nStaples) {
    if (thid % 2 == 0) {
      sMem[thid] = pdFt[nPID] / pdMOI[nPID] + pdFt[nPID + nStride] / pdMOI[nPID + nStride];
    }
    else {
      sMem[thid] = double(pnContacts[nPID] + pnContacts[nPID + nStride]);
    }
    __syncthreads();

    nStride /= 2;
    while (nStride > 32) 
      if (thid < nStride)
	sMem[thid] += sMem[thid + nStride];
      nStride /= 2;
      __syncthreads();
    }
	  
    if (thid < 32) {
      sMem[thid] += sMem[thid + 32];
      if (thid < 16) {
	sMem[thid] += sMem[thid + 16];
	if (thid < 8) {
	  sMem[thid] += sMem[thid + 8];
	  if (thid < 4) {
	    sMem[thid] += sMem[thid + 4];
	    if (thid < 2) {
	      sMem[thid] += sMem[thid + 2];
	      if (thid == 0) {
	        float ftot = atomicAdd(dAvgW, float(sMem[0] / nStaples));
		int ntot = atomicAdd(nTotC, int(sMem[1]+0.5));
	      }
	    }
	  }
	}
      }
    }
    
}


////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////
void Staple_Box::strain_step(long unsigned int tTime, bool bSvStress, bool bSvPos, bool bSaveF)
{
  if (bSvStress)
    {
      cudaMemset((void *) d_pfSE, 0, 4*sizeof(float));
      cudaMemset((void *) d_pfAvgAngVelo, 0, sizeof(float));
      cudaMemset((void *) d_pnTotContacts, 0, sizeof(int));

      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 1> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE>>>
	    (m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	     m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	     d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	     d_pdFx, d_pdFy, d_pdFt, d_pnContacts, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 1> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE>>>
	    (m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	     m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	     d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	     d_pdFx, d_pdFy, d_pdFt, d_pnContacts, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions, calculating stresses");

      avg_contact_velo <<<m_nGridSize, m_nBlockSize, (16+m_nBlockSize)*sizeof(double)>>>
	(m_nStaples, d_pdFt, d_pnContacts, d_pdMOI, d_pfAvgAngVelo, d_pnTotContacts);

      cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
      if (bSvPos)
	{
	  cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
	}
      else if (bSaveF) {
	cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
      }
      cudaThreadSynchronize();
      checkCudaError("Averaging angular velocity, summing contacts");

      cudaMemcpy(&m_pfAvgAngVelo, d_pfAvgAngVelo, sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&m_pnTotContacts, d_pnTotContacts, sizeof(int), cudaMemcpyDeviceToHost);
      //cudaMemcpy(h_pnContacts, d_pnContacts, sizeof(int)*m_nStaples, cudaMemcpyDeviceToHost);
      //int nTotContacts = 0;
      //for (int s = 0; s < m_nStaples; s++) {
      //nTotContacts += h_pnContacts[s];
      //printf("%d ", h_pnContacts[s]);
      //}
      //printf("\n%d %d\n", nTotContacts, m_pnTotContacts); 
    }
  else
    {
      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 0> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcF>>>
	    (m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	     m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	     d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	     d_pdFx, d_pdFy, d_pdFt, d_pnContacts, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 0> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcF>>>
	    (m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	     m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	     d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	     d_pdFx, d_pdFy, d_pdFt, d_pnContacts, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions");
    }

  split_staple_2_sph <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, 
     d_pdR, d_pdAspn, d_pdAbrb, d_pdCOM, d_pdSpX, d_pdSpY, d_pdSpPhi);
  cudaThreadSynchronize();
  checkCudaError("Splitting staples to spherocylinder coordinates");

  switch (m_ePotential)
    {
    case HARMONIC:
      heun_corr <HARMONIC> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcF>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	 m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	 d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY);
      break;
    case HERTZIAN:
      heun_corr <HERTZIAN> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcF>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	 m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	 d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY);
    }

  cudaThreadSynchronize();
  checkCudaError("Updating estimates, moving particles");

  split_staple_2_sph <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, m_dGamma, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAspn, d_pdAbrb, 
     d_pdCOM, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
  cudaThreadSynchronize();
  checkCudaError("Splitting staples to spherocylinder coordinates");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
  if (bSvStress)
    {
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	      tTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
      if (bSvPos) {
	save_positions(tTime);
	if (bSaveF)
	  save_staple_forces(tTime);
	//save_spherocyl_positions(tTime);
      }
      else if (bSaveF) {
	//cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
	save_staple_forces(tTime);
      }
    } 

  m_dGamma += m_dStep * m_dStrainRate;
  m_dTotalGamma += m_dStep * m_dStrainRate;
  cudaThreadSynchronize();

  if (m_dGamma > 0.5)
    set_back_gamma();
  else if (*h_bNewNbrs)
    find_neighbors();
}


/////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////
void Staple_Box::save_positions(long unsigned int nTime)
{
  char szBuf[150];
  sprintf(szBuf, "%s/sd%010lu.dat", m_strDataDir.c_str(), nTime);
  const char *szFilePos = szBuf;
  FILE *pOutfPos;
  pOutfPos = fopen(szFilePos, "w");
  if (pOutfPos == NULL)
    {
      fprintf(stderr, "Could not open file for writing");
      exit(1);
    }

  int i = h_pnMemID[0];
  fprintf(pOutfPos, "%d %f %f %f %.14g %.14g %.14g %.14g %.14g %.14g %.14g\n", 
	  0, h_pdR[i], h_pdAspn[i], h_pdAbrb[i], h_pdX[i], h_pdY[i], h_pdPhi[i], m_dL, m_dPacking, m_dGamma, m_dTotalGamma);
  for (int p = 1; p < m_nStaples; p++)
    {
      i = h_pnMemID[p];
      fprintf(pOutfPos, "%d %f %f %f %.14g %.14g %.14g\n",
	      p, h_pdR[i], h_pdAspn[i], h_pdAbrb[i], h_pdX[i], h_pdY[i], h_pdPhi[i]);
    }

  fclose(pOutfPos); 
}

void Staple_Box::save_staple_forces(long unsigned int nTime)
{
  char szBuf[150];
  sprintf(szBuf, "%s/fs%010lu.dat", m_strDataDir.c_str(), nTime);
  const char *szFileF = szBuf;
  FILE *pOutfF;
  pOutfF = fopen(szFileF, "w");
  if (pOutfF == NULL)
    {
      fprintf(stderr, "Could not open file for writing");
      exit(1);
    }

  //cudaMemcpy(h_pdDtCoeffSin, d_pdDtCoeffSin, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_pdDtCoeffCos, d_pdDtCoeffCos, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdFx, d_pdFx, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdFy, d_pdFy, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdFt, d_pdFt, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < m_nStaples; i++) {
    double dSinP = sin(h_pdPhi[i]);
    double dCosP = cos(h_pdPhi[i]);
    double dDis = m_dStrainRate*(h_pdDtCoeffSin[i]*dSinP*dSinP + h_pdDtCoeffCos[i]*dCosP*dCosP);
    fprintf(pOutfF, "%.14g %.14g %.14g %.14g\n", h_pdFx[i], h_pdFy[i], h_pdFt[i], h_pdFt[i]/h_pdMOI[i] - dDis);
  }
  fclose(pOutfF);
}

void Staple_Box::save_spherocyl_positions(long unsigned int nTime)
{
  char szBuf[150];
  sprintf(szBuf, "%s/sp%010lu.dat", m_strDataDir.c_str(), nTime);
  const char *szFilePos = szBuf;
  FILE *pOutfPos;
  pOutfPos = fopen(szFilePos, "w");
  if (pOutfPos == NULL)
    {
      fprintf(stderr, "Could not open file for writing");
      exit(1);
    }

  double *h_pdSpX = (double*) malloc(sizeof(double)*3*m_nStaples); 
  double *h_pdSpY = (double*) malloc(sizeof(double)*3*m_nStaples);
  double *h_pdSpPhi = (double*) malloc(sizeof(double)*3*m_nStaples);
  double *h_pdSpR = (double*) malloc(sizeof(double)*3*m_nStaples); 
  double *h_pdSpA = (double*) malloc(sizeof(double)*3*m_nStaples);
  cudaMemcpy(h_pdPhi, d_pdPhi, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpX, d_pdSpX, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpY, d_pdSpY, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpPhi, d_pdSpPhi, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpR, d_pdSpR, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pdSpA, d_pdSpA, 3*m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);

  int i = 0;
  fprintf(pOutfPos, "%d %f %f %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n", 
	  0, h_pdSpR[i], h_pdSpA[i], h_pdSpX[i], h_pdSpY[i], h_pdSpPhi[i], m_dL, m_dPacking, m_dGamma, m_dTotalGamma);
  for (int p = 1; p < 3*m_nStaples; p++)
    {
      i = p;
      fprintf(pOutfPos, "%d %f %f %.12g %.12g %.12g\n",
	      p, h_pdSpR[i], h_pdSpA[i], h_pdSpX[i], h_pdSpY[i], h_pdSpPhi[i]);
    }

  fclose(pOutfPos);
  free(h_pdSpX); free(h_pdSpY); free(h_pdSpPhi);
  free(h_pdSpA); free(h_pdSpR);
}


////////////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////////////
void Staple_Box::run_strain(double dStartGamma, double dStopGamma, double dSvStressGamma, double dSvPosGamma, bool bSaveF)
{
  if (m_dStrainRate == 0.0)
    {
      fprintf(stderr, "Cannot strain with zero strain rate\n");
      exit(1);
    }

  printf("Beginnig strain run with strain rate: %g and step %g\n", m_dStrainRate, m_dStep);
  fflush(stdout);

  if (dSvStressGamma < m_dStrainRate * m_dStep)
    dSvStressGamma = m_dStrainRate * m_dStep;
  if (dSvPosGamma < m_dStrainRate)
    dSvPosGamma = m_dStrainRate;

  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = (unsigned long)(dStartGamma / m_dStrainRate + 0.5);
  unsigned long int nStop = (unsigned long)(dStopGamma / m_dStrainRate + 0.5);
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned int nSvStressInterval = (unsigned int)(dSvStressGamma / (m_dStrainRate * m_dStep) + 0.5);
  unsigned int nSvPosInterval = (unsigned int)(dSvPosGamma / m_dStrainRate + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Strain run configured\n");
  printf("Start: %lu, Stop: %lu, Int step: %lu\n", nTime, nStop, nIntStep);
  printf("Stress save int: %lu, Pos save int: %lu\n", nSvStressInterval, nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atoi(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);

	    }
	}
    }

  sprintf(szBuf, "%s/sd_angvelo_contact.dat", m_strDataDir.c_str());
  const char *szPathWC = szBuf;
  FILE *outfWC;
  if (nTime == 0) {     
    outfWC = fopen(szPathWC, "w");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else {
    outfWC = fopen(szPathWC, "r+");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
    
    int nTpos = 0;
    while (nTpos != nTime)
      {
	if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	  {
	    int nPos = strcspn(szBuf, " ");
	    char szTime[20];
	    strncpy(szTime, szBuf, nPos);
	    szTime[nPos] = '\0';
	    nTpos = atoi(szTime);
	  }
	else
	  {
	    fprintf(stderr, "Reached end of file without finding start position");
	    fclose(outfWC);
	    outfWC = fopen(szPathWC, "a");
	    break;
	  }
      }
  }

  if (bSaveF) {
    cudaMemcpy(h_pdDtCoeffSin, d_pdDtCoeffSin, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pdDtCoeffCos, d_pdDtCoeffCos, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  }

  // Run strain for specified number of steps
  while (nTime < nStop)
    {
      bool bSvPos = (nTime % nSvPosInterval == 0);
      if (bSvPos) {
	strain_step(nTime, 1, 1, bSaveF);
	fprintf(outfWC, "%d %.7g %d\n", nTime, m_pfAvgAngVelo, m_pnTotContacts);
      }
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  strain_step(nTime, bSvStress, 0);
	  if (bSvStress)
	    fprintf(outfWC, "%d %.7g %d\n", nTime, m_pfAvgAngVelo, m_pnTotContacts);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  strain_step(nTime, bSvStress, 0);
	  if (bSvStress)
	    fprintf(outfWC, "%d %.7g %d\n", nTime, m_pfAvgAngVelo, m_pnTotContacts);
	  nTotalStep += 1;
	}
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  strain_step(nTime, 1, 1, bSaveF);
  //calculate_stress_energy();
  //cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  //cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  //cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  //cudaThreadSynchronize();
  //m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  //fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
  //	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  //fprintf(outfWC, "%d %.7g %d\n", nTime, m_pfAvgAngVelo, m_pnTotContacts);
  //save_positions(nTime);
  //save_spherocyl_positions(nTime);
  
  fclose(m_pOutfSE);
}

void Staple_Box::run_strain(long unsigned int nSteps)
{
  // Run strain for specified number of steps
  long unsigned int nTime = 0;
  while (nTime < nSteps)
    {
      strain_step(nTime, 0, 0);
      nTime += 1;
    }

}

/*
//////////////////////////////////////////////////////////////////////
void Staple_Box::run_shear_rate_loop(double dStartGammaDot, double dMaxGammaDot, double dMinGammaDot, double dRateOfChange, double dGammaWait, int nSvStressT, int nSvPosT)
{
  m_dStrainRate = dStartGammaDot;
  printf("Beginnig shear rate loop with rate: %g and step %g\n", m_dStrainRate, m_dStep);
  fflush(stdout);

  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned int nSvStressInterval = nSvStressT;
  unsigned int nSvPosInterval = nSvPosT;
  
  printf("Strain run configured\n");
  printf("Int step: %lu\n", nIntStep);
  printf("Stress save int: %lu, Pos save int: %lu\n", nSvStressInterval, nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  unsigned int nTime == 0;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL)
    {
      fprintf(stderr, "Could not open file for writing");
      exit(1);
    }
  
  // Run strain for specified number of steps
  int nTotalStep = 0;
  while (m_dStrainRate > dMinGammaDot)
    {
      int nShearT = int(dGammaWait / m_dStrainRate + 0.5);
      for (int s = 0; s < nShearT; s++) {
	bool bSvPos = (nTime % nSvPosInterval == 0);
	if (bSvPos)
	  strain_step(nTime, 1, 1);
	else
	  {
	    bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	    strain_step(nTime, bSvStress, 0);
	  }
	nTotalStep += 1;
	for (unsigned int nI = 1; nI < nIntStep; nI++)
	  {
	    bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	    strain_step(nTime, bSvStress, 0);
	    nTotalStep += 1;
	  }
	nTime += 1;
	//if (nTime % nReorderInterval == 0)
	//reorder_particles();
      }
      m_dStrainRate /= dRateOfChange;
    }
  while (m_dStrainRate < dMaxGammaDot)
    {
      int nShearT = int(dGammaWait / m_dStrainRate + 0.5);
      for (int s = 0; s < nShearT; s++) {
	bool bSvPos = (nTime % nSvPosInterval == 0);
	if (bSvPos)
	  strain_step(nTime, 1, 1);
	else
	  {
	    bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	    strain_step(nTime, bSvStress, 0);
	  }
	nTotalStep += 1;
	for (unsigned int nI = 1; nI < nIntStep; nI++)
	  {
	    bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	    strain_step(nTime, bSvStress, 0);
	    nTotalStep += 1;
	  }
	nTime += 1;
	//if (nTime % nReorderInterval == 0)
	//reorder_particles();
      }
      m_dStrainRate *= dRateOfChange;
    }
  
  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  save_positions(nTime);
  save_spherocyl_positions(nTime);
  
  fclose(m_pOutfSE);
}
*/


template<Potential ePot>
__global__ void euler_est(int nStaples, int *pnNPP, int *pnNbrList, double dL, double dGamma, 
			  double dStrain, double dStep, double *pdX, double *pdY, double *pdPhi, 
			  double *pdR, double *pdA, double *pdCOM, double *pdMOI, double *pdSinCoef, 
			  double *pdCosCoef, double *pdXcom, double *pdYcom, double *pdPhicom, 
			  double *pdFx, double *pdFy, double *pdFt, int *pnContacts, 
			  double *pdSE, double *pdTempX, double *pdTempY, double *pdTempPhi)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  int offset = 4 * blockDim.x / 3 + 8; // +8 should help to avoid a few bank conflicts
  sData[thid] = 0.0;
  sData[blockDim.x+thid] = 0.0;
  sData[2*blockDim.x+thid] = 0.0;
  sData[3*blockDim.x+thid] = 0.0;
  for (int i = thid; i < 4*offset; i += blockDim.x)
    sData[4*blockDim.x + i] = 0.0;
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
	  dDeltaX += dL * (double(dDeltaX < -0.5*dL) - double(dDeltaX > 0.5*dL));
	  dDeltaY += dL * (double(dDeltaY < -0.5*dL) - double(dDeltaY > 0.5*dL));
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
	      sData[3*blockDim.x+thid] += 1.0;
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
	      dDeltaXcom += dL * (double(dDeltaXcom < -0.5*dL) - double(dDeltaXcom > 0.5*dL));
	      dDeltaYcom += dL * (double(dDeltaYcom < -0.5*dL) - double(dDeltaYcom > 0.5*dL));
	      dDeltaXcom += dGamma * dDeltaYcom + s*nxA;
	      dDeltaYcom += s*nyA;
	      dFt += dDeltaXcom * dPfy - dDeltaYcom * dPfx;
	      if (nAdjPID > nPID)
		{
		  sData[4*blockDim.x + thid] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * nStaples);
		  sData[4*blockDim.x + thid + offset] += dPfx * dDx / (dL * dL);
		  sData[4*blockDim.x + thid + 2*offset] += dPfy * dDy / (dL * dL);
		  sData[4*blockDim.x + thid + 3*offset] += dPfx * dDy / (dL * dL);
		} 
	    }
	}
      sData[thid] = dFx;
      sData[blockDim.x + thid] = dFy;
      sData[2*blockDim.x + thid] = dFt;
      __syncthreads();
      int nF = thid % 3;
      int nSt = thid / 3;
      int b = nF*blockDim.x + 3*nSt;
      sData[b] += sData[b + 1] + sData[b + 2];
      switch (nF) 
	{
	case 0:
	  pdFx[nPID / 3] = sData[b];
	  pnContacts[nPID / 3] = sData[3*blockDim.x+thid] 
	    + sData[3*blockDim.x+thid+1] + sData[3*blockDim.x+thid+2];
	  break;
	case 1:
	  pdFy[nPID / 3] = sData[b];
	  break;
	case 2:
	  pdFt[nPID / 3] = sData[b];
	}
      __syncthreads();
      
      if (nF == 0) {
	dFx = sData[thid];
	dFy = sData[blockDim.x + thid];
	dFt = sData[2*blockDim.x + thid] / pdMOI[nPID/3];
	
	pdTempX[nPID / 3] = pdXcom[nPID / 3] + dStep * (dFx - dGamma * dFy);
	pdTempY[nPID / 3] = pdYcom[nPID / 3] + dStep * dFy;
	double dPhiC = pdPhicom[nPID / 3];
	double dCosPhi = cos(dPhiC);
	double dSinPhi = sin(dPhiC);
	double dFtDis = pdCosCoef[nPID/3] * dCosPhi * dCosPhi 
	  + pdSinCoef[nPID/3] * dSinPhi * dSinPhi;
	pdTempPhi[nPID / 3] = dPhiC + dStep * (dFt - dStrain * dFtDis);
      }
      
      nPID += nThreads;
    }

  __syncthreads();
    
  // Now we do a parallel reduction sum to find the total number of contacts
  int stride = 2*blockDim.x / 3;
  int base = 4*blockDim.x + thid;
  if (thid < stride) {
    while (base < 4*blockDim.x + 4*offset) {
      sData[base] += sData[base + stride];
      base += offset;
    }
  }
  stride /= 2; // stride is 1/4 block size, all threads perform 1 add
  __syncthreads();
  
  base = 4*blockDim.x + thid % stride + offset * (thid / stride);
  if (thid < 2*stride) {
    sData[base] += sData[base + stride];
    base += 2*offset;
    sData[base] += sData[base + stride];
  }
  stride /= 2;
  __syncthreads();
  
  while (stride > 8) {
    if (thid < 4 * stride) {
      base = 4*blockDim.x + thid % stride + offset * (thid / stride);
      sData[base] += sData[base + stride];
    }
    stride /= 2;  
    __syncthreads();
  }
  
  if (thid < 32) { //unroll end of loop
    base = 4*blockDim.x + thid % 8 + offset * (thid / 8);
    sData[base] += sData[base + 8];
    if (thid < 16) {
      base = 4*blockDim.x + thid % 4 + offset * (thid / 4);
      sData[base] += sData[base + 4];
      if (thid < 8) {
	base = 4*blockDim.x + thid % 2 + offset * (thid / 2);
	sData[base] += sData[base + 2];
	if (thid < 4) {
	  sData[4*blockDim.x + thid * offset] += sData[4*blockDim.x + thid * offset + 1];
	  //float tot = atomicAdd(pfSE+thid, (float)sData[3*blockDim.x + thid*offset]);
	  pdSE[blockIdx.x + thid*gridDim.x] = sData[4*blockDim.x + thid*offset];
	}
      }
    }
  }  
  
}

__global__ void sum_stress(double *pdBlockSE, double *pdSE)
{
  extern __shared__ double sData[];
  sData[threadIdx.x] = pdBlockSE[threadIdx.x];
  int nBlockSize = blockDim.x / 4;
  int nStride = nBlockSize / 2;
  int nBlock = threadIdx.x / nStride;
  int thid = threadIdx.x % nStride;
  __syncthreads();
  
  while (nStride > 8) {
    int ai = nBlock * nBlockSize + thid;
    int bi = ai + nStride;
    if (nBlock < 4)
      sData[ai] += sData[bi];
    nStride /= 2;
    nBlock = threadIdx.x / nStride;
    thid = threadIdx.x % nStride;
    __syncthreads();
  }
  if (nStride == 8) {
    nBlock = threadIdx.x / 8;
    thid = threadIdx.x % 8;
    int ai = nBlock * nBlockSize + thid;
    int bi = ai + 8;
    if (nBlock < 4)
      sData[ai] += sData[bi];
  }
  nBlock = threadIdx.x / 4;
  thid = threadIdx.x % 4;
  int ai = nBlock * nBlockSize + thid;
  int bi = ai + 4;
  if (nBlock < 4)
    sData[ai] += sData[bi];
  nBlock = threadIdx.x / 2;
  thid = threadIdx.x % 2;
  ai = nBlock * nBlockSize + thid;
  bi = ai + 2;
  if (nBlock < 4)
    sData[ai] += sData[bi];
  if (threadIdx.x < 4) {
    sData[threadIdx.x * nBlockSize] += sData[threadIdx.x*nBlockSize + 1];
    pdSE[threadIdx.x] = sData[threadIdx.x * nBlockSize];
  }
  
}


void Staple_Box::relax_step(long unsigned int tTime, bool bSvStress, bool bSvPos)
{
  cudaMemset((void *) d_pdBlockSE, 0, 4*sizeof(double)*m_nSpGridSize);
  cudaMemset((void *) d_pdSE, 0, 4*sizeof(double));
  cudaMemset((void *) d_pfAvgAngVelo, 0, sizeof(float));
  cudaMemset((void *) d_pnTotContacts, 0, sizeof(int));
  //cudaMemset((void *) d_pfSE, 0, 4*sizeof(float));

  switch (m_ePotential)
    {
    case HARMONIC:
      euler_est <HARMONIC> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, m_dStep, 
	 d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, d_pdMOI, 
	 d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, d_pdFx, d_pdFy, 
	 d_pdFt, d_pnContacts, d_pdBlockSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
      break;
    case HERTZIAN:
      euler_est <HERTZIAN> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcFSE>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, m_dStep, 
	 d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, d_pdMOI, 
	 d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, d_pdFx, d_pdFy, 
	 d_pdFt, d_pnContacts, d_pdBlockSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
    }
  cudaThreadSynchronize();
  checkCudaError("Estimating new particle positions, calculating stresses");
  
  int bs = 4*m_nSpGridSize;
  int sm = 4*m_nSpGridSize*sizeof(double);
  sum_stress <<<1, bs, sm>>> (d_pdBlockSE, d_pdSE);

  avg_contact_velo <<<m_nGridSize, m_nBlockSize, (16+m_nBlockSize)*sizeof(double)>>>
	(m_nStaples, d_pdFt, d_pnContacts, d_pdMOI, d_pfAvgAngVelo, d_pnTotContacts);

  //cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  if (bSvPos)
    {
      cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
    }

  split_staple_2_sph <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, 
     d_pdR, d_pdAspn, d_pdAbrb, d_pdCOM, d_pdSpX, d_pdSpY, d_pdSpPhi);

  cudaMemcpy(h_pnContacts, d_pnContacts, m_nStaples*sizeof(int), cudaMemcpyDeviceToHost);

  cudaThreadSynchronize();
  checkCudaError("Summing stresses; Splitting staples to spherocylinder coordinates");

  cudaMemcpyAsync(h_pdSE, d_pdSE, 4*sizeof(double), cudaMemcpyDeviceToHost);
  
  switch (m_ePotential)
    {
    case HARMONIC:
      heun_corr <HARMONIC> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcF>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	 m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	 d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY);
      break;
    case HERTZIAN:
      heun_corr <HERTZIAN> <<<m_nSpGridSize, m_nSpBlockSize, m_nSM_CalcF>>>
	(m_nStaples, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, m_dStrainRate, 
	 m_dStep, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdSpR, d_pdSpA, d_pdCOM, 
	 d_pdMOI, d_pdDtCoeffSin, d_pdDtCoeffCos, d_pdX, d_pdY, d_pdPhi, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY);
    }

  cudaMemcpy(&m_pfAvgAngVelo, d_pfAvgAngVelo, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&m_pnTotContacts, d_pnTotContacts, sizeof(int), cudaMemcpyDeviceToHost);

  cudaThreadSynchronize();
  checkCudaError("Updating estimates, moving particles");

  split_staple_2_sph <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, m_dGamma, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAspn, d_pdAbrb, 
     d_pdCOM, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
  cudaThreadSynchronize();
  checkCudaError("Splitting staples to spherocylinder coordinates");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
  if (bSvStress)
    {
      m_dP = 0.5 * (*m_pdPxx + *m_pdPyy);
      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	      tTime, *m_pdEnergy, *m_pdPxx, *m_pdPyy, m_dP, *m_pdPxy);
    }
  if (bSvPos) {
	save_positions(tTime);
	//save_spherocyl_positions(tTime);
  } 
  cudaThreadSynchronize();

  if (*h_bNewNbrs)
    find_neighbors();
}


__global__ void shrink(int nStaples, double dPctShrink, double *pdX, double *pdY)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  
  while(thid < nStaples) {
    pdX[thid] *= dPctShrink;
    pdY[thid] *= dPctShrink;
    thid += blockDim.x * gridDim.x;
  }
}


void Staple_Box::shrink_step(double dShrinkStep, FILE *pOutfSrk, FILE *pOutfAVC, bool bSave)
{
  m_dP = 0.5 * (*m_pdPxx + *m_pdPyy);
  //m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  if (bSave) {
    fprintf(pOutfSrk, "%.7g %.8g %.8g %.8g %.8g %.8g\n", 
	    m_dPacking, *m_pdEnergy, *m_pdPxx, *m_pdPyy, m_dP, *m_pdPxy);
    fprintf(pOutfAVC, "%.7g %.8g %d\n", m_dPacking, m_pfAvgAngVelo, m_pnTotContacts);
  }

  shrink <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, dShrinkStep, d_pdX, d_pdY);
  m_dL *= dShrinkStep;
  m_dPacking = calculate_packing();
  cudaThreadSynchronize();

  split_staple_2_sph <<<m_nGridSize, m_nBlockSize>>> 
    (m_nStaples, m_dGamma, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdAspn, d_pdAbrb, 
     d_pdCOM, d_pdSpX, d_pdSpY, d_pdSpPhi, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
  cudaThreadSynchronize();
  checkCudaError("Splitting staples to spherocylinder coordinates");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
  reconfigure_cells();
  cudaThreadSynchronize();

  if (*h_bNewNbrs)
    find_neighbors();
}


void Staple_Box::shrink_box(long unsigned int nStart, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStressInterval, unsigned int nSvPosInterval)
{
  m_dPacking = calculate_packing();
  printf("Beginning box shrink with packing fraction: %f\n", m_dPacking);
  
  
  double dStrainStep = m_dStep;
  m_dStep = dRelaxStep;
  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = nStart;
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Compression configured\n");
  printf("Start: %lu, Int step: %lu\n", nTime, nIntStep);
  printf("Pos save int: %lu\n", nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atoi(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }
  

  // Run strain for specified number of steps
  double dShrinkStep = 1. - dShrinkRate;
  printf("Shrink rate: %f, shrink step: %f\n", dShrinkRate, dShrinkStep);
  
  sprintf(szBuf, "%s/srk_stress_energy.dat", m_strDataDir.c_str());
  const char *szSrkSE = szBuf;
  FILE *pOutfSrk;
  if (nTime == 0)
    {
      pOutfSrk = fopen(szSrkSE, "w");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      pOutfSrk = fopen(szSrkSE, "r+");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      for (unsigned long int t = 0; t <= nTime; t++) {
	fgets(szBuf, 200, pOutfSrk);
      }
      fgets(szBuf, 200, pOutfSrk);
      int nPos = strcspn(szBuf, " ");
      char szPack[20];
      strncpy(szPack, szBuf, nPos);
      szPack[nPos] = '\0';
      double dPack = atof(szPack);
      if (dPack > (1 + dShrinkRate) * m_dPacking || dPack < (1 - dShrinkRate) * m_dPacking) {
	fprintf(stderr, "System packing fraction %g does not match with time %d (%g): ", m_dPacking, nTime, dPack);
	exit(1);
      }
    }
  
  sprintf(szBuf, "%s/sd_angvelo_contact.dat", m_strDataDir.c_str());
  const char *szPathWC = szBuf;
  FILE *outfWC;
  if (nTime == 0) {     
    outfWC = fopen(szPathWC, "w");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else {
    outfWC = fopen(szPathWC, "r+");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
    
    int nTpos = 0;
    while (nTpos != nTime)
      {
	if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	  {
	    int nPos = strcspn(szBuf, " ");
	    char szTime[20];
	    strncpy(szTime, szBuf, nPos);
	    szTime[nPos] = '\0';
	    nTpos = atoi(szTime);
	  }
	else
	  {
	    fprintf(stderr, "Reached end of file without finding start position");
	    fclose(outfWC);
	    outfWC = fopen(szPathWC, "a");
	    break;
	  }
      }
  }
 
  double dEMin = 1.0e-20;
  //double dDEMin = 1.0e-9;
  //double *h_pdBlockSE;
  //cudaHostAlloc((void**) &h_pdBlockSE, sizeof(double)*4*m_nGridSize, 0);
  while (m_dPacking < dFinalPacking)
    {
      shrink_step(dShrinkStep, pOutfSrk, outfWC);
      relax_step(nTime, 1, 0);
      nTotalStep += 1;
      bool bSvPos = (nTime % nSvPosInterval == 0);
      double dLastE = *m_pdEnergy;
      for (int r = 0; r < 10000000; r++) {
	if (*m_pdEnergy < dEMin) {
	  printf("Relaxation (step %d) stopped due to zero energy\n", nTime);
	  break;
	}
	relax_step(nTime,(r % nSvStressInterval == 0), 0);
	nTotalStep += 2;
	if (dLastE - *m_pdEnergy <= 0.0) {
	  for (int t = 0; t < 12; t++) {
	    relax_step(nTime, 0, 0);
	    if (*m_pdEnergy < dLastE) {
	      goto relax_continue;
	    }
	  }
	  printf("Relaxation (step %d) stopped due to small delta-E\n", nTime);
	  break;
	}
      relax_continue:
	dLastE = *m_pdEnergy;
	
	if (r == 9999999) {
	  printf("Relaxation (step %d) stopped due to long relaxation time\n", nTime);
	  m_dStep *= 1.25;
	}
      }
      relax_step(nTime, 1, bSvPos);
      //cudaMemcpyAsync(h_pdBlockSE, d_pdBlockSE, 4*m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
      //printf("Float Stress: %g %g %g %g\n", *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy);
      //printf("Double Stress: %g %g %g %g\n", *m_pdEnergy, *m_pdPxx, *m_pdPyy, *m_pdPxy);
      //printf("h_pdSE: %g %g %g %g\n", h_pdSE[0], h_pdSE[1], h_pdSE[2], h_pdSE[3]);
      //printf("Block sums:\n");
      //cudaThreadSynchronize();
      //for (int s = 0; s < 4; s++) {
      //	for (int b = 0; b < m_nGridSize; b++) {
      //	  printf("%g ", h_pdBlockSE[b]);
      //	}
      //	printf("\n");
      //}
      fflush(stdout);
      fflush(pOutfSrk);
      fflush(m_pOutfSE);
      nTotalStep += 1;
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  fclose(pOutfSrk);
  //cudaFreeHost(h_pdBlockSE);

  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  save_positions(nTime);
  //save_spherocyl_positions(nTime);
  m_dStep = dStrainStep;
  
  fclose(m_pOutfSE);
}







///////////////////////////////////////////////////////////////////////////////////
//
//
void Staple_Box::expand_box(long unsigned int nStart, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStressInterval, unsigned int nSvPosInterval)
{
  m_dPacking = calculate_packing();
  printf("Beginning box shrink with packing fraction: %f\n", m_dPacking);
  
  
  double dStrainStep = m_dStep;
  m_dStep = dRelaxStep;
  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = nStart;
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Compression configured\n");
  printf("Start: %lu, Int step: %lu\n", nTime, nIntStep);
  printf("Pos save int: %lu\n", nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atoi(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }

  // Run strain for specified number of steps
  double dShrinkStep = 1. + dShrinkRate;
  printf("Shrink rate: %f, shrink step: %f\n", -dShrinkRate, dShrinkStep);
  
  sprintf(szBuf, "%s/srk_stress_energy.dat", m_strDataDir.c_str());
  const char *szSrkSE = szBuf;
  FILE *pOutfSrk;
  if (nTime == 0)
    {
      pOutfSrk = fopen(szSrkSE, "w");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      pOutfSrk = fopen(szSrkSE, "r+");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      for (unsigned long int t = 0; t < nTime; t++) {
	fgets(szBuf, 200, pOutfSrk);
      }
      fgets(szBuf, 200, pOutfSrk);
      int nPos = strcspn(szBuf, " ");
      char szPack[20];
      strncpy(szPack, szBuf, nPos);
      szPack[nPos] = '\0';
      double dPack = atof(szPack);
      if (dPack > (1 + dShrinkRate) * m_dPacking || dPack < (1 - dShrinkRate) * m_dPacking) {
	fprintf(stderr, "System packing fraction %g does not match with time %d", dPack, nTime);
	exit(1);
      }
    }
 
  sprintf(szBuf, "%s/sd_angvelo_contact.dat", m_strDataDir.c_str());
  const char *szPathWC = szBuf;
  FILE *outfWC;
  if (nTime == 0) {     
    outfWC = fopen(szPathWC, "w");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else {
    outfWC = fopen(szPathWC, "r+");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
    
    int nTpos = 0;
    while (nTpos != nTime)
      {
	if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	  {
	    int nPos = strcspn(szBuf, " ");
	    char szTime[20];
	    strncpy(szTime, szBuf, nPos);
	    szTime[nPos] = '\0';
	    nTpos = atoi(szTime);
	  }
	else
	  {
	    fprintf(stderr, "Reached end of file without finding start position");
	    fclose(outfWC);
	    outfWC = fopen(szPathWC, "a");
	    break;
	  }
      }
  }

  float fEMin = 1.0e-20;
  float fDEMin = 1.0e-7;
  while (m_dPacking > dFinalPacking)
    {
      shrink_step(dShrinkStep, pOutfSrk, outfWC);
      relax_step(nTime, 1, 0);
      nTotalStep += 1;
      bool bSvPos = (nTime % nSvPosInterval == 0);
      float fLastE = *m_pfEnergy;
      for (int r = 0; r < 1000000; r++) {
	if (*m_pdEnergy < fEMin) {
	  printf("Relaxation (step %d) stopped due to zero energy\n", nTime);
	  break;
	}
	relax_step(nTime, 0, 0);
	relax_step(nTime, 0, 0);
	relax_step(nTime, !bool((r+nSvStressInterval/2)%nSvStressInterval), 0);
	nTotalStep += 2;
	if (fLastE - *m_pfEnergy <= fDEMin * (*m_pfEnergy)) {
	  printf("Relaxation (step %d) stopped due to small delta-E\n", nTime);
	  //if (*m_pdEnergy > dLastE) {
	  //  m_dStep *= 0.8;
	  //  printf("Reducing step size to: %g\n", m_dStep);
	  //}
	  break;
	}
	fLastE = *m_pfEnergy;
	if (r == 999999) {
	  printf("Relaxation (step %d) stopped due to long relaxation time\n", nTime);
	  m_dStep *= 1.25;
	}
      }
      relax_step(nTime, 1, bSvPos);
      fflush(stdout);
      fflush(pOutfSrk);
      fflush(m_pOutfSE);
      nTotalStep += 1;
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  fclose(pOutfSrk);

  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  save_positions(nTime);
  //save_spherocyl_positions(nTime);
  m_dStep = dStrainStep;
  
  fclose(m_pOutfSE);
}





///////////////////////////////////////////////////////////
//
//
void Staple_Box::simple_shrink_box(long unsigned int nSteps, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStressInterval, unsigned int nSvPosInterval)
{
  m_dPacking = calculate_packing();
  printf("Beginning box shrink with packing fraction: %f\n", m_dPacking);
  
  
  double dStrainStep = m_dStep;
  m_dStep = dRelaxStep;
  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = 0;
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Compression configured\n");
  printf("Start: %lu, Int step: %lu\n", nTime, nIntStep);
  printf("Pos save int: %lu\n", nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atoi(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }

  // Run strain for specified number of steps
  double dShrinkStep = 1. - dShrinkRate;
  printf("Shrink rate: %f, shrink step: %f\n", dShrinkRate, dShrinkStep);
  
  sprintf(szBuf, "%s/srk_stress_energy.dat", m_strDataDir.c_str());
  const char *szSrkSE = szBuf;
  FILE *pOutfSrk;
  if (nTime == 0)
    {
      pOutfSrk = fopen(szSrkSE, "w");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      pOutfSrk = fopen(szSrkSE, "r+");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      for (unsigned long int t = 0; t < nTime; t++) {
	fgets(szBuf, 200, pOutfSrk);
      }
      fgets(szBuf, 200, pOutfSrk);
      int nPos = strcspn(szBuf, " ");
      char szPack[20];
      strncpy(szPack, szBuf, nPos);
      szPack[nPos] = '\0';
      double dPack = atof(szPack);
      if (dPack > (1 + dShrinkRate) * m_dPacking || dPack < (1 - dShrinkRate) * m_dPacking) {
	fprintf(stderr, "System packing fraction %g does not match with time %d", dPack, nTime);
	exit(1);
      }
    }

  sprintf(szBuf, "%s/sd_angvelo_contact.dat", m_strDataDir.c_str());
  const char *szPathWC = szBuf;
  FILE *outfWC;
  if (nTime == 0) {     
    outfWC = fopen(szPathWC, "w");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else {
    outfWC = fopen(szPathWC, "r+");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
    
    int nTpos = 0;
    while (nTpos != nTime)
      {
	if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	  {
	    int nPos = strcspn(szBuf, " ");
	    char szTime[20];
	    strncpy(szTime, szBuf, nPos);
	    szTime[nPos] = '\0';
	    nTpos = atoi(szTime);
	  }
	else
	  {
	    fprintf(stderr, "Reached end of file without finding start position");
	    fclose(outfWC);
	    outfWC = fopen(szPathWC, "a");
	    break;
	  }
      }
  }
 
  float dEMin = 1e-20;
  //double *h_pdBlockSE;
  //cudaHostAlloc((void**) &h_pdBlockSE, sizeof(double)*4*m_nGridSize, 0);
  while (m_dPacking < dFinalPacking)
    {
      shrink_step(dShrinkStep, pOutfSrk, outfWC);
      for (int r = 0; r < nSteps; r++) {
	relax_step(nTime, (r % nSvStressInterval == 0), 0);
	nTotalStep += 1;
	if (*m_pdEnergy < dEMin) {
	  printf("Compression step %d stopped after %d relaxation steps due to zero energy\n", nTime, r+1);
	  break;
	}
	else if (r == nSteps - 1) {
	  printf("Compression step %d ended after %d relaxation steps\n", nTime, r+1);
	}
      }
      bool bSvPos = (nTime % nSvPosInterval == 0);
      relax_step(nTime, 1, bSvPos);
      //cudaMemcpyAsync(h_pdBlockSE, d_pdBlockSE, 4*m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
      //printf("Float Stress: %g %g %g %g\n", *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy);
      //printf("Double Stress: %g %g %g %g\n", *m_pdEnergy, *m_pdPxx, *m_pdPyy, *m_pdPxy);
      //printf("h_pdSE: %g %g %g %g\n", h_pdSE[0], h_pdSE[1], h_pdSE[2], h_pdSE[3]);
      //printf("Block sums:\n");
      //cudaThreadSynchronize();
      //for (int s = 0; s < 4; s++) {
      //	for (int b = 0; b < m_nGridSize; b++) {
      //	  printf("%g ", h_pdBlockSE[b]);
      //	}
      //	printf("\n");
      //}
      fflush(stdout);
      fflush(pOutfSrk);
      fflush(m_pOutfSE);
      nTotalStep += 1;
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  fclose(pOutfSrk);
  //cudaFreeHost(h_pdBlockSE);

  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  save_positions(nTime);
  //save_spherocyl_positions(nTime);
  m_dStep = dStrainStep;
  
  fclose(m_pOutfSE);
}


long unsigned int Staple_Box::simplest_shrink_box(long unsigned int nTime, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStressInterval, unsigned int nSvPosInterval, bool bAppend)
{
  m_dPacking = calculate_packing();
  printf("Beginning box shrink with packing fraction: %f\n", m_dPacking);
  
  
  double dStrainStep = m_dStep;
  m_dStep = dRelaxStep;
  // +0.5 to cast to nearest integer rather than rounding down
  //unsigned long int nTime = nTime;
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Compression configured\n");
  printf("Start: %lu, Int step: %lu\n", nTime, nIntStep);
  printf("Pos save int: %lu\n", nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else if (bAppend) {
    m_pOutfSE = fopen(szPathSE, "a");
    if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
  }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atoi(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }
  
  sprintf(szBuf, "%s/srk_stress_energy.dat", m_strDataDir.c_str());
  const char *szSrkSE = szBuf;
  FILE *pOutfSrk;
  if (nTime == 0)
    {
      pOutfSrk = fopen(szSrkSE, "w");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else if (bAppend)
    {
      pOutfSrk = fopen(szSrkSE, "a");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      pOutfSrk = fopen(szSrkSE, "r+");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}

      double dPack = 0.0;
      while (dPack < m_dPacking)
	{
	  if (fgets(szBuf, 200, pOutfSrk) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szPack[20];
	      strncpy(szPack, szBuf, nPos);
	      szPack[nPos] = '\0';
	      dPack = atof(szPack);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start packing %g", m_dPacking);
	      exit(1);
	    }
	}
    }
  
  sprintf(szBuf, "%s/sd_angvelo_contact.dat", m_strDataDir.c_str());
  const char *szPathWC = szBuf;
  FILE *outfWC;
  if (nTime == 0) {     
    outfWC = fopen(szPathWC, "w");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else if (bAppend) {     
    outfWC = fopen(szPathWC, "a");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else {
    outfWC = fopen(szPathWC, "r+");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }

    double dPack = 0.0;
    while (dPack < m_dPacking)
      {
	if (fgets(szBuf, 200, outfWC) != NULL)
	  {
	    int nPos = strcspn(szBuf, " ");
	    char szPack[20];
	    strncpy(szPack, szBuf, nPos);
	    szPack[nPos] = '\0';
	    dPack = atof(szPack);
	  }
	else
	  {
	    fprintf(stderr, "Reached end of file without finding start position");
	    fclose(outfWC);
	    //outfWC = fopen(szPathWC, "a");
	    //break;
	    exit(1);
	  }
      }
  }
 
  
  // Run strain for specified number of steps
  double dShrinkStep = pow(1. - dShrinkRate, m_dStep);
  printf("Shrink rate: %f, shrink step: %f\n", dShrinkRate, dShrinkStep);
  int nSaveRate = int(1e-5/dShrinkRate + 0.5);
  //double *h_pdBlockSE;
  //cudaHostAlloc((void**) &h_pdBlockSE, sizeof(double)*4*m_nGridSize, 0);
  while (m_dPacking < dFinalPacking)
    {
      
      for (int i = 0; i < nIntStep; i++) {
	bool bSave = (nTotalStep % nSaveRate == 0);
	shrink_step(dShrinkStep, pOutfSrk, outfWC, bSave);
	bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	relax_step(nTime, bSvStress, 0);
	nTotalStep += 1;
      }
      bool bSvStress = (nTotalStep % nSvStressInterval == 0);
      bool bSave = (nTotalStep % nSaveRate == 0);
      bool bSvPos = (nTime % nSvPosInterval == 0);
      shrink_step(dShrinkStep, pOutfSrk, outfWC, bSave);
      relax_step(nTime, bSvStress, bSvPos);

      fflush(stdout);
      fflush(pOutfSrk);
      fflush(m_pOutfSE);
      fflush(outfWC);
      nTotalStep += 1;
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  fclose(pOutfSrk);
  //cudaFreeHost(h_pdBlockSE);

  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  save_positions(nTime);
  //save_spherocyl_positions(nTime);
  m_dStep = dStrainStep;
  
  fclose(m_pOutfSE);
  return nTime;
}


long unsigned int Staple_Box::simplest_expand_box(long unsigned int nTime, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStressInterval, unsigned int nSvPosInterval, bool bAppend)
{
  m_dPacking = calculate_packing();
  printf("Beginning box expansion with packing fraction: %f\n", m_dPacking);
  
  
  double dStrainStep = m_dStep;
  m_dStep = dRelaxStep;
  // +0.5 to cast to nearest integer rather than rounding down
  //unsigned long int nTime = nTime;
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Compression configured\n");
  printf("Start: %lu, Int step: %lu\n", nTime, nIntStep);
  printf("Pos save int: %lu\n", nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else if (bAppend)
    {
      m_pOutfSE = fopen(szPathSE, "a");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atoi(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }
  
  sprintf(szBuf, "%s/srk_stress_energy.dat", m_strDataDir.c_str());
  const char *szSrkSE = szBuf;
  FILE *pOutfSrk;
  if (nTime == 0)
    {
      pOutfSrk = fopen(szSrkSE, "w");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else if (bAppend)
    {
      pOutfSrk = fopen(szSrkSE, "a");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      pOutfSrk = fopen(szSrkSE, "r+");
      if (pOutfSrk == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}

      double dPack = 0.0;
      while (dPack < m_dPacking)
	{
	  if (fgets(szBuf, 200, pOutfSrk) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szPack[20];
	      strncpy(szPack, szBuf, nPos);
	      szPack[nPos] = '\0';
	      dPack = atof(szPack);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start packing %g", m_dPacking);
	      exit(1);
	    }
	}
    }
  
  sprintf(szBuf, "%s/sd_angvelo_contact.dat", m_strDataDir.c_str());
  const char *szPathWC = szBuf;
  FILE *outfWC;
  if (nTime == 0) {     
    outfWC = fopen(szPathWC, "w");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else if (bAppend) {     
    outfWC = fopen(szPathWC, "a");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }
  }
  else {
    outfWC = fopen(szPathWC, "r+");
    if (outfWC == NULL)
      {
	fprintf(stderr, "Could not open file for writing");
	exit(1);
      }

    double dPack = 0.0;
    while (dPack < m_dPacking)
      {
	if (fgets(szBuf, 200, outfWC) != NULL)
	  {
	    int nPos = strcspn(szBuf, " ");
	    char szPack[20];
	    strncpy(szPack, szBuf, nPos);
	    szPack[nPos] = '\0';
	    dPack = atof(szPack);
	  }
	else
	  {
	    fprintf(stderr, "Reached end of file without finding start position");
	    fclose(outfWC);
	    //outfWC = fopen(szPathWC, "a");
	    //break;
	    exit(1);
	  }
      }
  }
 
  
  // Run strain for specified number of steps
  double dShrinkStep = 1.0 / pow(1. - dShrinkRate, m_dStep);
  printf("Shrink rate: %f, shrink step: %f\n", dShrinkRate, dShrinkStep);
  int nSaveRate = int(1e-5/dShrinkRate + 0.5);
  //double *h_pdBlockSE;
  //cudaHostAlloc((void**) &h_pdBlockSE, sizeof(double)*4*m_nGridSize, 0);
  while (m_dPacking > dFinalPacking)
    {
      
      for (int i = 0; i < nIntStep; i++) {
	bool bSave = (nTotalStep % nSaveRate == 0);
	shrink_step(dShrinkStep, pOutfSrk, outfWC, bSave);
	bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	relax_step(nTime, bSvStress, 0);
	nTotalStep += 1;
      }
      bool bSvStress = (nTotalStep % nSvStressInterval == 0);
      bool bSave = (nTotalStep % nSaveRate == 0);
      bool bSvPos = (nTime % nSvPosInterval == 0);
      shrink_step(dShrinkStep, pOutfSrk, outfWC, bSave);
      relax_step(nTime, bSvStress, bSvPos);

      fflush(stdout);
      fflush(pOutfSrk);
      fflush(m_pOutfSE);
      fflush(outfWC);
      nTotalStep += 1;
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  fclose(pOutfSrk);
  //cudaFreeHost(h_pdBlockSE);

  // Save final configuration
  calculate_stress_energy();
  cudaMemcpyAsync(h_pdX, d_pdX, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nStaples*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
  fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  save_positions(nTime);
  //save_spherocyl_positions(nTime);
  m_dStep = dStrainStep;
  
  fclose(m_pOutfSE);

  return nTime;
}


long int Staple_Box::relax_box(long int nStartTime, int nMaxSteps, double dStepSize, int nMinimumSteps, int nStressSaveStep)
{
  double dStrainRate = m_dStrainRate;
  m_dStrainRate = dStrainRate;
  double dStep = m_dStep;
  m_dStep = dStepSize;
  long int nTime = nStartTime;
  
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_strDataDir.c_str(), m_strFileSE.c_str());
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "a");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  sprintf(szBuf, "%s/relaxed_se.dat", m_strDataDir.c_str());
  const char *szPathRSE = szBuf;
  FILE *pOutfRSE;
  if (nTime == 0)
    {
      pOutfRSE = fopen(szPathRSE, "w");
      if (pOutfRSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      pOutfRSE = fopen(szPathRSE, "a");
      if (pOutfRSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  
  
  calculate_stress_energy();
  cudaMemcpy(h_pfSE, d_pfSE, 4*sizeof(float), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  m_fP = 0.5*(*m_pfPxx + *m_pfPyy);
  fprintf(pOutfRSE, "%.7g %lu %.7g %.7g %.7g %.7g %.7g ", 
	  m_dTotalGamma, nTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy);
  double dEMin = *m_pfEnergy; 
  double dECutoff = 1e-20;
  int tMin = nTime;
  while (nTime < nStartTime + nMaxSteps) {
    bool bSvStress = (nTime % nStressSaveStep == 0);
    relax_step(nTime, bSvStress, 0);
    nTime += 1;
    if (*m_pdEnergy < dECutoff) {
      m_dP = 0.5 * (*m_pdPxx + *m_pdPyy);
      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pdEnergy, *m_pdPxx, *m_pdPyy, m_dP, *m_pdPxy);
      printf("Relaxation stopped due to zero energy\n");
      break;
    }
    if (*m_pdEnergy < dEMin) {
      dEMin = *m_pdEnergy;
      tMin = nTime;
    }
    else {
      int tDiff = nTime - tMin;
      if (tDiff > nMinimumSteps) {
	m_dP = 0.5 * (*m_pdPxx + *m_pdPyy);
	fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pdEnergy, *m_pdPxx, *m_pdPyy, m_dP, *m_pdPxy);
	printf("Relaxation stopped due to local minimum\n");
	break;
      }
    }
  }

  m_dP = 0.5*(*m_pdPxx + *m_pdPyy);
  fprintf(pOutfRSE, "%lu %.7g %.7g %.7g %.7g %.7g\n", 
	  nTime, *m_pdEnergy, *m_pdPxx, *m_pdPyy, m_dP, *m_pdPxy);

  fclose(pOutfRSE);
  fclose(m_pOutfSE);
  
  m_dStrainRate = dStrainRate;
  m_dStep = dStep;

  return nTime;
}
