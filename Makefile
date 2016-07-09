CXX=g++
CC=gcc
LD=g++ 
#CXX=/usr/local/bin/g++
#CC= /usr/local/bin/gcc
#LD= /usr/local/bin/g++ 
SSE2_FLAGS=
SSE3_FLAGS=
AVX_FLAGS=
FAST_DIVISION_FLAG=
OPENMP_FLAGS= -DOPENMP -fopenmp
LOPENMP= -lgomp
OPT_FLAGS= -O3 -funroll-all-loops
64BIT_FLAGS= -m64
ifdef 32BIT
64BIT_FLAGS=
endif
ifdef DEBUG
 OPT_FLAGS= -ggdb
endif
ifdef SINGLE_THREADED
 OPENMP_FLAGS=
 LOPENMP=
endif
ifdef FAST_DIVISION
 FAST_DIVISION_FLAG= -DFAST_DIVISION
endif

ifdef SSE2
 SSE2_FLAGS = -DSSE2 -msse -msse2
endif
ifdef SSE3
 SSE3_FLAGS = -DSSE2 -DSSE3 -msse3
endif
ifdef AVX
 AVX_FLAGS = -mavx -DSSE2 -DSSE3 -DAVX 
endif


ifdef GPU
 ifdef AMD
  OPENCLINCLUDE= /opt/AMDAPP/include
  ifdef 32BIT
   OPENCLLIB=  /opt/AMDAPP/lib/x86
  else
   OPENCLLIB=  /opt/AMDAPP/lib/x86_64
  endif
  CFLAGS= -I./ -I$(OPENCLINCLUDE) -I$(OPENCLLIB) $(OPT_FLAGS) $(FAST_DIVISION_FLAG) $(SSE2_FLAGS) $(SSE3_FLAGS) $(AVX_FLAGS) $(OPENMP_FLAGS) $(64BIT_FLAGS) -DGPU -DAMD -fexpensive-optimizations -ffast-math -finline-functions -frerun-loop-opt -static $(HDRS) $(DEFINES)
 endif
 ifdef NVIDIA  
  OPENCLINCLUDE= /usr/local/cuda/include
  ifdef 32BIT
   OPENCLLIB= /usr/lib
  else
  OPENCLLIB= /usr/lib64
  CFLAGS= -I./ -I$(OPENCLINCLUDE) -I$(OPENCLLIB) $(OPT_FLAGS) $(FAST_DIVISION_FLAG) $(SSE2_FLAGS) $(SSE3_FLAGS) $(AVX_FLAGS) $(OPENMP_FLAGS) $(64BIT_FLAGS) -DGPU -DNVIDIA -fexpensive-optimizations -ffast-math -finline-functions -frerun-loop-opt -static $(HDRS) $(DEFINES)
  endif
 endif
 LIBS= -lOpenCL -lm -lc -lgcc -lrt $(LOPENMP)
else
 CFLAGS= -I./  $(OPT_FLAGS) $(FAST_DIVISION_FLAG) $(SSE2_FLAGS) $(SSE3_FLAGS) $(AVX_FLAGS) $(OPENMP_FLAGS) $(64BIT_FLAGS) -fexpensive-optimizations -ffast-math -finline-functions -frerun-loop-opt -static $(HDRS) $(DEFINES)
 LIBS= -lm -lc -lgcc -lrt $(LOPENMP)
endif

# object files needed
ifdef GPU
OBJ_fast_protein_cluster = fast_protein_cluster.o\
        error_handlers.o\
	libtmscore_cpu.o\
	libtmscore_gpu.o
fast_protein_cluster : $(OBJ_fast_protein_cluster)
	$(LD) -L$(OPENCLLIB)  -o fast_protein_cluster $(LFLAGS) $(OBJ_fast_protein_cluster) $(LIBS)
	cp tmscore.cl $(OPENCLINCLUDE)/tmscore.cl
	cp rmsd.cl $(OPENCLINCLUDE)/rmsd.cl
else
OBJ_fast_protein_cluster = fast_protein_cluster.o\
        error_handlers.o\
	libtmscore_cpu.o
fast_protein_cluster : $(OBJ_fast_protein_cluster)
	$(LD) -o fast_protein_cluster $(LFLAGS) $(OBJ_fast_protein_cluster) $(LIBS)
endif
libtmscore_cpu.o: libtmscore_cpu.cpp
	$(CXX) $(CFLAGS) -c libtmscore_cpu.cpp
error_handlers.o: error_handlers.c
	$(CC) $(CFLAGS) -c error_handlers.c
fast_protein_cluster.o: fast_protein_cluster.cpp
	$(CXX) $(CFLAGS) -c fast_protein_cluster.cpp
ifdef GPU
 libtmscore_gpu.o: libtmscore_gpu.cpp
	$(CXX) $(CFLAGS) -c libtmscore_gpu.cpp
endif

clean:
	rm *.o
	rm core.*

# all together now...
default: fast_protein_cluster
