# modal_gemm.py
import modal

app = modal.App("cuda-gemm")

image = (
    modal.Image.debian_slim()
    .apt_install("build-essential")
    .pip_install()  # 不需要 python 包
)

@app.function(
    gpu="any",          # 自动分配 GPU
    timeout=600,
    cpu=2,
    memory=8192
)
def run_cuda():
    import subprocess
    subprocess.run(["nvcc", "gemm.cu", "-O2", "-o", "gemm"], check=True)
    subprocess.run(["./gemm"], check=True)

@app.local_entrypoint()
def main():
    run_cuda.remote()