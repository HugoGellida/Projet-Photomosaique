# Compilateur CUDA
NVCC = nvcc -ccbin /usr/bin/gcc-10

# Nom de l'exécutable
TARGET = bin/app

# Dossiers
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Récupération récursive des sources
CPP_SRC := $(shell find $(SRC_DIR) -name "*.cpp")
CU_SRC  := $(shell find $(SRC_DIR) -name "*.cu")
SRC     := $(CPP_SRC) $(CU_SRC)

# Génération des fichiers objets
OBJ := $(patsubst $(SRC_DIR)/%, $(BUILD_DIR)/%, $(SRC:.cpp=.o))
OBJ := $(OBJ:.cu=.o)

# Options de compilation
CFLAGS = -O2 -std=c++17 -I$(INCLUDE_DIR)
HOSTFLAGS = -Xcompiler -Wno-unused-result
CUDAFLAGS = -arch=sm_60

# Règle principale
all: $(TARGET)

# Link final
$(TARGET): $(OBJ)
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(HOSTFLAGS) $^ -o $@ -lstdc++ -lstdc++fs -lm

# Compilation des .cpp
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(HOSTFLAGS) -c $< -o $@

# Compilation des .cu
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(HOSTFLAGS) -c $< -o $@

# Nettoyage
clean:
	find $(BUILD_DIR) -name "*.o" -delete
	rm -f $(TARGET)