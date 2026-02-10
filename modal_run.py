import modal

app = modal.App("neu-info5100-oak-spr-2025")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "cmake")
)

@app.function(
    gpu="H100",
    timeout=600,  # 增加到10分钟，以便运行所有 benchmark
    cpu=4,
    memory=16384,
    image=image,
)
def run():
    import os
    import subprocess

    def sh(cmd: str, check: bool = True):
        print(f"\n$ {cmd}")
        subprocess.run(["bash", "-lc", cmd], check=check)

    # 1) 打印 GPU / CUDA 工具链信息
    sh("nvidia-smi")
    sh("nvcc --version")

    # 2) 拉取文章对应仓库
    if not os.path.exists("SGEMM_CUDA"):
        sh("git clone https://github.com/siboehm/SGEMM_CUDA.git")

    sh("ls -la")
    sh("cd SGEMM_CUDA && ls -la")

    # 3) 编译
    sh("cd SGEMM_CUDA && (make clean || true)")
    sh("cd SGEMM_CUDA && make -j")

    # 4) 查看编译输出的文件
    sh("cd SGEMM_CUDA && find . -name 'sgemm' -o -name '*benchmark*' -o -type f -executable")

    # 5) 运行 benchmark
    print("\n" + "="*60)
    print("Running SGEMM Benchmark - All Kernels")
    print("="*60)
    sh("cd SGEMM_CUDA/build && ./sgemm", check=True)

    print("\n✅ Done. Copy nvidia-smi + benchmark output into your README/results.")

@app.local_entrypoint()
def main():
    run.remote()