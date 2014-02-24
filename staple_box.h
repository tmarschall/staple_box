/*  staple_box.h
 *
 *  Arrays with the prefix "h_" are allocated on the host (CPU), 
 *   those with the prefix "d_" are allocated on the device (GPU).
 *  Anything starting with "g_" is solely used in "gold" functions 
 *   on the cpu (for error checking) and will only be compiled if 
 *   the environmental variable GOLD_FUNCS=1
 *  
 */

#ifndef GOLD_FUNCS
#define GOLD_FUNCS 0
#endif

#ifndef GRANULAR_BOX_H
#define GRANULAR_BOX_H

#include <cstdio>
#include <cstring>
#include <string>

enum Potential {HARMONIC = 2, HERTZIAN = 5};

class Staple_Box
{
 private:
  int m_nStaples;
  double m_dL;  // Box side length
  double m_dPacking;  // Packing fraction
  double m_dGamma;   // Strain parameter
  double m_dTotalGamma;
  Potential m_ePotential;  // Soft core interaction (harmonic or hertzian)
  double m_dStrainRate;
  double m_dStep;

  // file output data
  FILE *m_pOutfSE;  // output steam
  std::string m_strDataDir;  // directory to save data (particle configurations)
  std::string m_strFileSE;  // filename for stress energy data

  // Coordinates etc. for staples
  double m_dRMax;  // Largest radius
  double m_dAMax;  // Largest spherocylinder spine
  // Position and orientation arrays to be allocated on the CPU (host)
  double *h_pdX;
  double *h_pdY;
  double *h_pdPhi;
  double *h_pdR;  // Radii
  double *h_pdAspn;
  double *h_pdAbrb;
  double *h_pdMOI;
  double *h_pdDtCoeffSin;
  double *h_pdDtCoeffCos;
  int *h_pnMemID;  // IDs of particles reordered in memory
  int *h_bNewNbrs;
  // Position and orientation arrays to be allocated on the GPU (device)
  double *d_pdX;
  double *d_pdY;
  double *d_pdPhi;
  double *d_pdR;
  double *d_pdAspn;
  double *d_pdAbrb;
  double *d_pdMOI;  // Moment of Inertia
  double *d_pdDtCoeffSin;  // Coeff on the sin term of the angular dissipation
  double *d_pdDtCoeffCos;  // coeff on the cos term
  double *d_pdTempX;  // Temporary positions used in several routines
  double *d_pdTempY;  //  only needed on the device
  double *d_pdTempPhi;
  int *d_pnInitID;  // The original ID of the particle
  int *d_pnMemID;  // ID of the current location of the particle in memory
  int *d_bNewNbrs;  // Set to 1 when particle moves more than dEpsilon
  // Arrays for "gold" routines (for testing)
#if GOLD_FUNCS == 1
  double *g_pdX;
  double *g_pdY;
  double *g_pdPhi;
  double *g_pdR;
  double *g_pdAspn;
  double *g_pdAbrb;
  double *g_pdMOI;  
  double *g_pdDtCoeffSin;  
  double *g_pdDtCoeffCos; 
  double *g_pdTempX;
  double *g_pdTempY;
  double *g_pdTempPhi;
  int *g_pnInitID;
  int *g_pnMemID;
  double *g_pdXMoved;
  double *g_pdYMoved;
  int *g_bNewNbrs;
#endif

  // Constituent spherocylinder coordinates
  double *d_pdSpX;
  double *d_pdSpY;
  double *d_pdSpPhi;
  double *d_pdSpR;
  double *d_pdSpA;
  double *d_pdXMoved;  // Amount each spherocylinder moved since last finding neighbors
  double *d_pdYMoved;
  double *d_pdCOM;
  bool m_bCOM;  // Set to zero if the com offsets need to be calculated
#if GOLD_FUNCS == 1
  double *g_pdSpX;
  double *g_pdSpY;
  double *g_pdSpPhi;
  double *g_pdSpR;
  double *g_pdSpA;
  double *g_pdCOM;
#endif

  // Stresses and forces
  float *m_pfEnergy;
  float *m_pfPxx;
  float *m_pfPyy;
  float *m_pfPxy;
  float m_fP;  // Total pressure
  float *h_pfSE;  // Array for transfering the stresses and energy
  float m_pfAvgAngVelo;
  int m_pnTotContacts;
  double *m_pdEnergy;
  double *m_pdPxx;
  double *m_pdPyy;
  double *m_pdPxy;
  double m_dP;
  double *h_pdSE;
  double *h_pdFx;
  double *h_pdFy;
  double *h_pdFt;
  int *h_pnContacts;
  // GPU
  float *d_pfSE;
  float *d_pfAvgAngVelo;
  int *d_pnTotContacts;
  double *d_pdSE;
  double *d_pdBlockSE;
  double *d_pdFx;
  double *d_pdFy;
  double *d_pdFt;
  int *d_pnContacts;
#if GOLD_FUNCS == 1
  float *g_pfSE;
  double *g_pdFx;
  double *g_pdFy;
  double *g_pdFt;
#endif
  
  // These variables are used for spatial subdivision and
  // neighbor lists for faster contact detection etc.
  int m_nCells; 
  int m_nCellRows;
  int m_nCellCols;
  double m_dCellW;    // Cell width
  double m_dCellH;    // Cell height
  int *h_pnCellID;    // Cell ID for each particle
  int *h_pnAdjCells;  // Which cells are next to each other
  int m_nMaxPPC;      // Max particles per cell
  int *h_pnPPC;       // Number of particles in each cell
  int *h_pnCellList;  // List of particles in each cell
  int m_nMaxNbrs;     // Max neighbors a particle can have
  int *h_pnNPP;       // Number of neighbors (possible contacts) of each particle
  int *h_pnNbrList;   // List of each particle's neighbors
  // GPU arrays
  int *d_pnCellID;
  int *d_pnAdjCells;
  int *d_pnPPC;
  int *d_pnCellList;
  int *d_pnNPP;
  int *d_pnNbrList;
#if GOLD_FUNCS == 1
  int *g_pnCellID;
  int *g_pnAdjCells;
  int *g_pnPPC;
  int *g_pnCellList;
  int *g_pnNPP;
  int *g_pnNbrList;
#endif

  // Used to not have to update neighbors every step when things are moving
  double m_dEpsilon;

  // These variables define configurations for kernel cuda kernel launches
  int m_nGridSize;  // Grid size (# of thread blocks) for finding cell IDs
  int m_nBlockSize;  // Block size (# threads per block)
  int m_nSpGridSize;
  int m_nSpBlockSize;
  int m_nSM_CalcF;
  int m_nSM_CalcFSE;  // Shared memory per block
  int m_nDeviceMem;
  
  void construct_defaults();
  void reconfigure_cells();
  void configure_cells();
  void set_kernel_configs();
  double calculate_packing();
  
  void save_positions(long unsigned int nTime);
  void save_spherocyl_positions(long unsigned int nTime);
  void strain_step(long unsigned int nTime, bool bSvStress = 0, bool bSvPos = 0, bool bSvF = 0);
  void shrink_step(double dShrinkStep, FILE *pOutfSrk, FILE *pOutfAVC, bool bSave = 1);
  void relax_step(long unsigned int nTime, bool bSvStress = 0, bool bSvPos = 0);

 public:
  Staple_Box(int nStaples, double dL, double dRMax, double dAMax, double dEpsilon = 0.1,  
	       int nMaxPPC = 15, int nMaxNbrs = 25, Potential ePotential = HARMONIC);
  Staple_Box(int nStaples, double dL, double *pdX, double *pdY, double *pdPhi, double *pdR, 
	       double *pdAspn, double *dAbrb, double dEpsilon = 0.1,  int nMaxPPC = 16, 
	       int nMaxNbrs = 36, Potential ePotential = HARMONIC);
  ~Staple_Box();

  void place_random_staples(int seed = 0);
  void split_staples();
  void find_neighbors();
  void find_neighbors_2();
  void set_back_gamma();
  void reorder_particles();
  void reset_IDs();
  void calculate_stress_energy();
  bool check_for_contacts();
  bool check_for_contacts(int nIndex);
  void run_strain(double dStartGam, double dStopGam, double dSvStressGam, double dSvPosGam, bool bSaveF = 0);
  void run_strain(long unsigned int nSteps);
  void shrink_box(long unsigned int nStart, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStress, unsigned int nSvPos);
  void simple_shrink_box(long unsigned int nStart, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStress, unsigned int nSvPos);
  long unsigned int simplest_shrink_box(long unsigned int nTime, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStress, unsigned int nSvPos, bool bAppend = 0);
  void expand_box(long unsigned int nStart, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStress, unsigned int nSvPos);
  long unsigned int simplest_expand_box(long unsigned int nTime, double dShrinkRate, double dRelaxStep, double dFinalPacking, unsigned int nSvStress, unsigned int nSvPos, bool bAppend = 0);
  void run_shear_rate_loop(double dMaxGammaDot, double dMinGammaDot, double dRateOfChange, int nSvStressT, int nSvPosT);
  long int relax_box(long int nStartTime, int nMaxSteps, double dStepSize, int nMinimumSteps, int nStressSaveStep);
  
#if GOLD_FUNCS == 1
  void calculate_stress_energy_gold();
  void compare_calculate_stress_energy();
  void split_staples_gold();
  void compare_split_staples();
  void find_neighbors_gold();
  void compare_find_neighbors();
  void set_back_gamma_gold();
  void reorder_particles_gold();
  void compare_reorder_particles();
#endif

  void display(bool bParticles = 1, bool bCells = 1, bool bNbrs = 1, bool bStress = 1);
  
  void save_staple_forces(long unsigned int nTime);

  void set_gamma(double dGamma) { m_dGamma = dGamma; }
  void set_total_gamma(double dTotalGamma) { m_dTotalGamma = dTotalGamma; }
  void set_step(double dStep) { m_dStep = dStep; }
  void set_strain(double dStrain) { m_dStrainRate = dStrain; }
  void set_data_dir(std::string strDataDir) { m_strDataDir = strDataDir; }
  void set_se_file(std::string strFileSE) { m_strFileSE = strFileSE; }

  double* getFx() { return h_pdFx; }
  double* getFy() { return h_pdFy; }
  double* getFt() { return h_pdFt; }
  double* getMOI() { return h_pdMOI; }
  
};

#endif
