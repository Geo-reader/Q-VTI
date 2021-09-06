#! /bin/sh

MPICH2_HOME=/home/wangn/mpi
CUDA_HOME=/usr/local/cuda
#SU_HOME=/home/chenhm/softwares/su


NVCC = $(CUDA_HOME)/bin/nvcc
MPICC = $(MPICH2_HOME)/bin/mpicc

#INC = $(CUDA_HOME)/include
INC= -I$(MPICH2_HOME)/include -I$(CUDA_HOME)/include
# -I$(SU_HOME)/include -I$(SU_HOME)/src/Complex/include


#LIB = -lcudart -lcurand -lcufft
LIB=-L$(CUDA_HOME)/lib64 -L$(MPICH2_HOME)/lib 
#-L$(SU_HOME)/lib


LINK= -lcudart -lcufft -lm -lmpich -lpthread -lrt -DMPICH_IGNORE_CXX_SEEK  -DMPICH_SKIP_MPICXX

CFILES = viscoelas_3D.cpp
CUFILES =cuviscoelas_3D.cu
OBJECTS = viscoelas_3D.o cuviscoelas_3D.o 
EXECNAME = a.out

all:
	$(MPICC) -w -c $(CFILES) $(INC) $(LIB) $(LINK) 
	$(NVCC) -w -c $(CUFILES) $(INC) $(LIB) $(LINK) 
	$(MPICC) -o $(EXECNAME) $(OBJECTS) $(INC) $(LIB) $(LINK) 

	rm -f *.o 
	nohup ./a.out&



	
