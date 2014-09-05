.SUFFIXES: .cu .cuh .h

INC = -I. -I./include  -I./include/nvidia_gpucomputingsdk_4.2.9_c_common_inc
LIBDIR = -L/usr/local/packages/cuda/4.2.9/cuda/lib64 -L/usr/local/packages/hdf5/1.8.10/Intel-13.0.0-openmpi-1.6.2/lib
#LIBDIR = -L/usr/local/packages/cuda/5.5.22/lib64 -L/usr/local/packages/hdf5/1.8.10/Intel-13.0.0-openmpi-1.6.2/lib
LIB = -lcudart #-lhdf5 #-lcutil_x86_64
#-lm

CC = h5pcc
CPP = h5pcc
CXX = nvcc
MPICC = h5pcc

# -v
#ARCHFLAG = -gencode arch=compute_20,code=sm_20
ARCHFLAG= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
OPTFLAG0 = -O0 -g -G
OPTFLAG1 = -O0
OPTFLAG2 = -O2
OPTFLAG3 = -O3
ASMFLAG = -ptx
PROFFLAG = --ptxas-options=-v -keep
#WFLAGS = -Wall # not supported by nvcc

CFLAGS = $(INC) -std=c99
CPPFLAGS = $(INC)
CXXFLAGS = $(INC) $(LIB) $(LIBDIR) $(ARCHFLAG)

srcdir = src
gpusrc = kernel.cu
cpusrc = host_main.cpp host_func.cpp host_launcher.cu
exec = mpi_ising #postprocess_ea postprocess_ea2 postprocess_q postprocess_qq postprocess_r12

default: $(exec) 


#.cu: Makefile
#	$(CXX) $(CFLAGS) -o $@ $*.cu
#


%.o: %.c
	$(CC) $(OPTFLAG2) $(CFLAGS) -c $<

%.o: %.cpp
	$(CPP) $(OPTFLAG2) $(CPPFLAGS) -c $<

%.o: %.cu
	$(CXX) $(OPTFLAG2) $(CXXFLAGS) -c $<

mpiprocess.o: mpiprocess.cpp
	$(MPICC) -c $<

mpi_ising: host_main.o mpiprocess.o host_func.o host_launcher.o kernel.o
	$(MPICC) $(OPTFLAG2) $(LIB) $(LIBDIR) -o $@ $^

ising: host_main.o host_func.o host_launcher.o kernel.o
	$(CXX) $(OPTFLAG2) $(CXXFLAGS) -o $@ $^

postprocess_%: postprocess_%.c
	$(CPP) -O2 $(CPPFLAGS) -o $@ $<



prof: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(PROFFLAG) $^

profclean: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(PROFFLAG) -clean $^

g_ising: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG0) $(CXXFLAGS) -o $@ $^

#ptx: $(src)
#	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(ASMFLAG) $^



install:
	cp $(exec) /work/sfeng/ising/short_l10/

clean:
	rm -r *.o $(exec)

cleanoutput:
	rm -r output_*

