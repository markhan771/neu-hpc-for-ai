# Makefile for CUDA GEMM Assignment
# Author: INFO7335 Week 02
# Date: February 2026

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O2 -arch=sm_70

# Target executable
TARGET = gemm

# Source files
SRC = gemm.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Help message
help:
	@echo "CUDA GEMM Makefile"
	@echo "=================="
	@echo "Usage:"
	@echo "  make          - Build the executable"
	@echo "  make run      - Build and run the program"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"

.PHONY: all run clean help
