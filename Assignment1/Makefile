# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2 -pthread
LDFLAGS = -pthread -lm

# Target executables
TARGETS = test benchmark

# Object files
OBJS_COMMON = matrix_mult_single.o matrix_mult_multi.o
OBJS_TEST = test.o $(OBJS_COMMON)
OBJS_BENCH = benchmark.o $(OBJS_COMMON)

# Default target
all: $(TARGETS)

# Build test executable
test: $(OBJS_TEST)
	$(CC) $(CFLAGS) -o $@ $(OBJS_TEST) $(LDFLAGS)

# Build benchmark executable
benchmark: $(OBJS_BENCH)
	$(CC) $(CFLAGS) -o $@ $(OBJS_BENCH) $(LDFLAGS)

# Compile object files
%.o: %.c matrix_mult.h
	$(CC) $(CFLAGS) -c $<

# Clean build artifacts
clean:
	rm -f *.o $(TARGETS)

# Run tests
run-test: test
	./test

# Run benchmark with default size (1000x1000)
run-benchmark: benchmark
	./benchmark

# Run benchmark with custom size (e.g., make run-benchmark-2048)
run-benchmark-%: benchmark
	./benchmark $*

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build all executables (default)"
	@echo "  test             - Build test executable"
	@echo "  benchmark        - Build benchmark executable"
	@echo "  clean            - Remove all build artifacts"
	@echo "  run-test         - Build and run tests"
	@echo "  run-benchmark    - Build and run benchmark (1000x1000)"
	@echo "  run-benchmark-N  - Build and run benchmark with NxN matrices"
	@echo "                     Example: make run-benchmark-2048"
	@echo "  help             - Show this help message"

.PHONY: all clean run-test run-benchmark help
