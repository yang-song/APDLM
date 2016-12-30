# Written by Alex Schwing.
#CC=The path to MPICXX
CC = /pkgs/mpich-3.0.4/bin/mpicxx

#CUDA_PATH=The path to cuda root, e.g., /pkgs_local/cuda-5.5
CUDA_PATH = $(CUDA_HOME)

CFLAGS	= -std=c++0x -pedantic -W -Wall -fopenmp -O3 -D_DEBUG -DWITH_MPI -fPIC -I$(CUDA_PATH)/include -msse3
CFLAGS_DEBUG	= -std=c++0x -pedantic -W -Wall -fopenmp -O3 -D_DEBUG -DWITH_MPI -fPIC -I$(CUDA_PATH)/include -msse3

NVCC = nvcc
NVCCFLAGS = -ccbin=$(CC) -std=c++11 -Xcompiler -fPIC -O3 -D_DEBUG -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35

LSDN_SRC = $(shell find libLSDN -name "*.cpp")
LSDN_CUSRC = $(shell find libLSDN -name "*.cu")
LSDN_H = $(shell find libLSDN -name "*.h")
GEN_H = $(shell find . -maxdepth 1 -name "*.h")

LSDN_SRC_NP = $(shell find libLSDN -name "*.cpp" | rev | cut -d "/" -f 1 | rev)
LSDN_OBJ = ${LSDN_SRC_NP:.cpp=.o}
LSDN_CUSRC_NP = $(shell find libLSDN -name "*.cu" | rev | cut -d "/" -f 1 | rev)
LSDN_CUOBJ = ${LSDN_CUSRC_NP:.cu=.cuo}

clean:
	rm -f *.o
	rm -f *.a
	rm -f *.cuo

%.o: libLSDN/%.cpp
	$(CC) $< -c -o $@ $(CFLAGS)

%.cuo: libLSDN/%.cu
	$(NVCC) $< -c -o $@ $(NVCCFLAGS)

libLSDN.a: $(LSDN_OBJ) $(LSDN_CUOBJ) $(LSDN_H) $(GEN_H)
	ar rcs libLSDN.a $(LSDN_OBJ) $(LSDN_CUOBJ)
