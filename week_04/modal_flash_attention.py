"""
Modal script to compile and run FlashAttention CUDA implementation on cloud GPU.

Usage:
    modal run modal_flash_attention.py
"""

import modal

app = modal.App("flash-attention")

# Define CUDA image with necessary tools
cuda_image = (
    modal.Image.debian_slim()
    .apt_install("build-essential")
    .run_commands(
        "apt-get update",
        "apt-get install -y wget",
        # Install CUDA toolkit
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-3",
    )
)


@app.function(
    gpu="T4",  # Use T4 GPU (you can change to A10G, A100, etc.)
    image=cuda_image,
    timeout=600,
)
def run_flash_attention():
    import subprocess
    import os

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the CUDA source file
    with open(os.path.join(script_dir, "flash_attention.cu"), "r") as f:
        cuda_code = f.read()

    # Write to temporary file
    with open("/tmp/flash_attention.cu", "w") as f:
        f.write(cuda_code)

    print("Compiling CUDA code...")
    compile_result = subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc",
            "-O3",
            "-arch=sm_75",  # T4 compute capability
            "/tmp/flash_attention.cu",
            "-o",
            "/tmp/flash_attention_cuda",
        ],
        capture_output=True,
        text=True,
    )

    if compile_result.returncode != 0:
        print("Compilation failed!")
        print("STDERR:", compile_result.stderr)
        return

    print("Compilation successful!\n")
    print("=" * 60)

    # Run the compiled program
    print("\nRunning FlashAttention CUDA kernel...\n")
    run_result = subprocess.run(
        ["/tmp/flash_attention_cuda"], capture_output=True, text=True
    )

    print(run_result.stdout)
    if run_result.stderr:
        print("STDERR:", run_result.stderr)


@app.local_entrypoint()
def main():
    """Entry point for modal run"""
    run_flash_attention.remote()
