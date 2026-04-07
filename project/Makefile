# Compilateur CUDA
NVCC = nvcc -ccbin /usr/bin/gcc-10

# Nom de l'exécutable
TARGET = main

# Fichiers sources
CPP_SRC := $(wildcard *.cpp)
CU_SRC := $(wildcard *.cu)
SRC = $(CPP_SRC) $(CU_SRC)

# Options de compilation
CFLAGS = -O2 -std=c++17
HOSTFLAGS =  -Xcompiler -Wno-unused-result
CUDAFLAGS = -arch=sm_60

# Règle principale
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(HOSTFLAGS) $(SRC) -o $(TARGET) -lstdc++ -lstdc++fs -lm

# Nettoyage
clean:
	rm -f $(TARGET)