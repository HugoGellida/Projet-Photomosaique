# Compilateur CUDA
NVCC = nvcc

# Nom de l'exécutable
TARGET = main

# Fichiers sources
SRC = main.cu ImageBase.cpp

# Options de compilation
CFLAGS = -O2
HOSTFLAGS =  -Xcompiler -Wno-unused-result
CUDAFLAGS = -arch=sm_60

# Règle principale
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(HOSTFLAGS) $(SRC) -o $(TARGET)

# Nettoyage
clean:
	rm -f $(TARGET)