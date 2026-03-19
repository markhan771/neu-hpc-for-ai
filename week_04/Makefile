# Makefile for FlashAttention implementations

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -lm
NVCCFLAGS = -O3 -arch=sm_70

# Targets
all: flash_attention_c flash_attention_cuda

# C implementation
flash_attention_c: flash_attention.c
	$(CC) $(CFLAGS) -o flash_attention_c flash_attention.c

# CUDA implementation
flash_attention_cuda: flash_attention.cu
	$(NVCC) $(NVCCFLAGS) -o flash_attention_cuda flash_attention.cu

# Run tests
test_c: flash_attention_c
	./flash_attention_c

test_cuda: flash_attention_cuda
	./flash_attention_cuda

test: test_c test_cuda

# Clean
clean:
	rm -f flash_attention_c flash_attention_cuda

.PHONY: all test test_c test_cuda clean
