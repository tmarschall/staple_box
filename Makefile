 # compilers
CXX=g++
NVCC=nvcc

# paths of cuda libraries
CUDAPATH=/usr/local/cuda/5.0/cuda
# for cudpp:
CUDASDKPATH=/usr/local/cuda/SDK/NVIDIA_GPU_Computing_SDK/C

# includes for header files
CXXINCLUDE=-I$(CUDAPATH)/include
CUDAINCLUDE=-I$(CUDAPATH)/include

# compiler flags: include all warnings, optimization level 3
CXXFLAGS=-O3 -DGOLD_FUNCS=0
NVCCFLAGS=--ptxas-options=-v -O3 -arch=sm_20 -DGOLD_FUNCS=0
# linker flags: include all warnings, include library files
LDFLAGS=-Wall -L$(CUDAPATH)/lib64 -lcudart

# object files
CXXOBJECTS=staple_box.o cudaErr.o dat_file_input.o
NVCCOBJECTS=find_neighbors.o calculate_stress_energy.o data_primitives_2.o strain_dynamics.o

# final executable
EXE=../../exe/staple_box
PROF_EXE=../../exe/staple_box_profile
SRK_EXE=../../exe/staple_srk
EXP_EXE=../../exe/staple_exp
SIMP_EXE=../../exe/staple_simple_srk
CYC_EXE=../../exe/staple_srk_cycle
CALC_EXE=../../exe/staple_calc_file
RELAX_EXE=../../exe/staple_relax
CHECK_EXE=../../exe/staple_check

#rules:

all: $(EXE) $(PROF_EXE) $(SRK_EXE)
sim: $(EXE)
srk: $(SRK_EXE)
exp: $(EXP_EXE)
simple: $(SIMP_EXE)
cycle: $(CYC_EXE)
prof: $(PROF_EXE)
calc: $(CALC_EXE)
relax: $(RELAX_EXE)
check: $(CHECK_EXE)

######################
# compile executable #
######################

$(EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main.o
ifeq ($(TEST), true)
	$(CXX) -o $(EXE)_test main.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
else
	$(CXX) -o $(EXE) main.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
endif

$(RELAX_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) relax_run.o
ifeq ($(TEST), true)
	$(CXX) -o $(RELAX_EXE)_test relax_run.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
else
	$(CXX) -o $(RELAX_EXE) relax_run.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
endif

$(SRK_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_srk.o
ifeq ($(TEST), true)
	$(CXX) -o $(SRK_EXE)_test main_srk.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
else	
	$(CXX) -o $(SRK_EXE) main_srk.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
endif

$(EXP_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_exp.o
ifeq ($(TEST), true)
	$(CXX) -o $(EXP_EXE)_test main_exp.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
else	
	$(CXX) -o $(EXP_EXE) main_exp.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
endif

$(SIMP_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_srk_simple.o
ifeq ($(TEST), true)
	$(CXX) -o $(SIMP_EXE)_test main_srk_simple.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
else
	$(CXX) -o $(SIMP_EXE) main_srk_simple.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
endif

$(CYC_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_srk_cycle.o
ifeq ($(TEST), true)
	$(CXX) -o $(CYC_EXE)_test main_srk_cycle.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
else
	$(CXX) -o $(CYC_EXE) main_srk_cycle.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)
endif

$(PROF_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_profile.o
	$(CXX) -o $(PROF_EXE) main_profile.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(CALC_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_calc_file.o
	$(CXX) -o $(CALC_EXE) main_calc_file.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(CHECK_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_check.o
	$(CXX) -o $(CHECK_EXE) main_check.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

###################
# compile objects #
###################

#c++ files
main.o: main.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main.cpp

main_profile.o: main_profile.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_profile.cpp

main_srk.o: main_srk.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_srk.cpp

main_exp.o: main_exp.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_exp.cpp

main_srk_simple.o: main_srk_simple.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_srk_simple.cpp

main_srk_cycle.o: main_srk_cycle.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_srk_cycle.cpp

main_calc_file.o: main_calc_file.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_calc_file.cpp

main_check.o: main_check.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_check.cpp

staple_box.o: staple_box.cpp staple_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c staple_box.cpp

cudaErr.o: cudaErr.h cudaErr.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c cudaErr.cpp

dat_file_input.o: file_input.h dat_file_input.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c dat_file_input.cpp

#cuda files
find_neighbors.o: staple_box.h data_primitives.h find_neighbors.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c find_neighbors.cu

calculate_stress_energy.o: staple_box.h calculate_stress_energy.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c calculate_stress_energy.cu

strain_dynamics.o: staple_box.h strain_dynamics.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c strain_dynamics.cu

data_primitives_2.o: data_primitives.h data_primitives_2.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c data_primitives_2.cu

#clean up object files or other assembly files
clean:
	rm -f *.o *.ptx

