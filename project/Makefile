# Compilateurs et flags
CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -Wall -O2
CXXFLAGS = -Wall -O2
NVCCFLAGS = -O2

# Dossiers
SRC_DIR = src
COMMON_DIR = common
OBJ_DIR = obj
BIN_DIR = bin

# Nom de l'exécutable
TARGET = $(BIN_DIR)/app

# Récupère tous les fichiers sources
C_SRCS = $(shell find $(SRC_DIR) $(COMMON_DIR) -type f -name "*.c")
CPP_SRCS = $(shell find $(SRC_DIR) $(COMMON_DIR) -type f -name "*.cpp")
CU_SRCS = $(shell find $(SRC_DIR) $(COMMON_DIR) -type f -name "*.cu")

# Transforme les fichiers sources en fichiers objets dans obj/
C_OBJS = $(patsubst %,$(OBJ_DIR)/%,$(C_SRCS:.c=.o))
CPP_OBJS = $(patsubst %,$(OBJ_DIR)/%,$(CPP_SRCS:.cpp=.o))
CU_OBJS = $(patsubst %,$(OBJ_DIR)/%,$(CU_SRCS:.cu=.o))

# Crée les sous-dossiers nécessaires dans obj
$(OBJ_DIR)/%:
	@mkdir -p $(dir $@)

# Règle par défaut
all: $(TARGET)

# Link
$(TARGET): $(C_OBJS) $(CPP_OBJS) $(CU_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

# Compilation des fichiers C
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Compilation des fichiers C++
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilation des fichiers CUDA
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Nettoyage
clean:
	rm -rf $(OBJ_DIR)/* $(BIN_DIR)/*