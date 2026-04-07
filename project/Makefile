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
HOSTFLAGS = -Xcompiler -Wno-unused-result
CUDAFLAGS = -arch=sm_60

# sfml
SFML_LIBS = -lsfml-graphics -lsfml-window -lsfml-system

SFML_INCLUDE = -I/path/to/SFML/include
SFML_LIBPATH = -L/path/to/SFML/lib

# Règle principale
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(HOSTFLAGS) $(SFML_INCLUDE) $(SRC) -o $(TARGET) \
	$(SFML_LIBPATH) $(SFML_LIBS) -lstdc++ -lstdc++fs -lm

# Nettoyage
clean:
	rm -f $(TARGET)