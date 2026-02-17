"""
Modal script to run CUDA SGEMM kernels on H100 GPU
Assignment 3: Replicate all code runs from https://siboehm.com/articles/22/CUDA-MMM
"""
import modal

app = modal.App("sgemm-cuda-h100")

# Configure CUDA environment for H100
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "cmake", "ninja-build")
    .pip_install("seaborn", "matplotlib", "pandas", "numpy")
)

@app.function(
    gpu="H100",
    timeout=1200,  # 20 minutes for all benchmarks
    cpu=8,
    memory=32768,  # 32GB RAM
    image=image,
)
def run_sgemm_benchmarks():
    """
    Clone, build, and run all SGEMM kernel benchmarks on H100 GPU
    """
    import os
    import subprocess
    import json
    from pathlib import Path

    def sh(cmd: str, check: bool = True, capture: bool = False):
        """Execute shell command"""
        print(f"\n$ {cmd}")
        result = subprocess.run(
            ["bash", "-c", cmd],
            check=check,
            capture_output=capture,
            text=True
        )
        if capture:
            return result.stdout
        return None

    # ========================================
    # 1. System Information
    # ========================================
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)

    sh("nvidia-smi")
    sh("nvcc --version")
    sh("lscpu | grep 'Model name'")

    # ========================================
    # 2. Clone Repository
    # ========================================
    print("\n" + "="*80)
    print("CLONING SGEMM_CUDA REPOSITORY")
    print("="*80)

    if not os.path.exists("SGEMM_CUDA"):
        sh("git clone https://github.com/siboehm/SGEMM_CUDA.git")

    os.chdir("SGEMM_CUDA")

    # ========================================
    # 3. Configure for H100 (Compute Capability 9.0)
    # ========================================
    print("\n" + "="*80)
    print("CONFIGURING FOR H100 (sm_90)")
    print("="*80)

    # Update CMakeLists.txt for H100
    cmake_content = Path("CMakeLists.txt").read_text()
    cmake_content = cmake_content.replace(
        "set(CUDA_COMPUTE_CAPABILITY 86)",
        "set(CUDA_COMPUTE_CAPABILITY 90)"
    )
    Path("CMakeLists.txt").write_text(cmake_content)
    print("✓ Updated compute capability to 90 (H100)")

    # ========================================
    # 4. Build
    # ========================================
    print("\n" + "="*80)
    print("BUILDING PROJECT")
    print("="*80)

    sh("mkdir -p build && cd build && cmake .. && cmake --build . -j8")

    # ========================================
    # 5. Run Individual Kernels
    # ========================================
    print("\n" + "="*80)
    print("RUNNING INDIVIDUAL KERNELS")
    print("="*80)

    kernels = {
        0: "cuBLAS (baseline)",
        1: "Naive",
        2: "Global Memory Coalescing",
        3: "Shared Memory Caching",
        4: "1D Blocktiling",
        5: "2D Blocktiling",
        6: "Vectorized Memory Access",
        7: "Bank Conflicts (Linearize)",
        8: "Bank Conflicts (Offset)",
        9: "Autotuning",
        10: "Warptiling",
        11: "Double Buffering",
    }

    results = {}
    for kernel_num, kernel_name in kernels.items():
        print(f"\n{'='*60}")
        print(f"Kernel {kernel_num}: {kernel_name}")
        print(f"{'='*60}")
        try:
            output = sh(f"cd build && ./sgemm {kernel_num}", capture=True)
            print(output)
            results[kernel_num] = {"name": kernel_name, "output": output}
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Kernel {kernel_num} failed: {e}")
            results[kernel_num] = {"name": kernel_name, "error": str(e)}

    # ========================================
    # 6. Run Full Benchmark Suite
    # ========================================
    print("\n" + "="*80)
    print("RUNNING FULL BENCHMARK SUITE")
    print("="*80)

    try:
        sh("cd build && ./sgemm")
    except:
        print("Full benchmark failed, but individual results captured")

    # ========================================
    # 7. Generate Plots (if script exists)
    # ========================================
    print("\n" + "="*80)
    print("GENERATING PERFORMANCE PLOTS")
    print("="*80)

    if os.path.exists("plot_benchmark_results.py"):
        try:
            sh("python plot_benchmark_results.py")
            print("✓ Plots generated")
        except:
            print("⚠️  Plot generation failed")

    # ========================================
    # 8. Summary
    # ========================================
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\nTotal kernels tested: {len(results)}")
    print(f"Successful: {sum(1 for r in results.values() if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in results.values() if 'error' in r)}")

    print("\n✅ Benchmark complete!")
    print("\nNext steps:")
    print("1. Review the output above")
    print("2. Copy GPU info and results to your README")
    print("3. Compare H100 results with the article's A6000 results")

    return results


@app.function(
    gpu="H100",
    timeout=300,
    cpu=4,
    memory=16384,
    image=image,
)
def run_single_kernel(kernel_num: int):
    """Run a single kernel for detailed analysis"""
    import subprocess
    import os

    def sh(cmd: str):
        print(f"\n$ {cmd}")
        subprocess.run(["bash", "-c", cmd], check=True)

    if not os.path.exists("SGEMM_CUDA"):
        sh("git clone https://github.com/siboehm/SGEMM_CUDA.git")

    os.chdir("SGEMM_CUDA")

    # Update for H100
    import pathlib
    cmake_content = pathlib.Path("CMakeLists.txt").read_text()
    cmake_content = cmake_content.replace(
        "set(CUDA_COMPUTE_CAPABILITY 86)",
        "set(CUDA_COMPUTE_CAPABILITY 90)"
    )
    pathlib.Path("CMakeLists.txt").write_text(cmake_content)

    # Build and run
    sh("mkdir -p build && cd build && cmake .. && cmake --build . -j4")
    sh(f"cd build && ./sgemm {kernel_num}")


@app.local_entrypoint()
def main(kernel: int = None):
    """
    Run SGEMM benchmarks on H100

    Args:
        kernel: Optional kernel number to run (0-11). If None, runs all kernels.
    """
    if kernel is not None:
        print(f"Running kernel {kernel} on H100...")
        run_single_kernel.remote(kernel)
    else:
        print("Running all kernels on H100...")
        results = run_sgemm_benchmarks.remote()
        print(f"\n✅ Completed! Results for {len(results)} kernels captured.")
