"""
Phase 0.1 sanity check.

Run this first on every fresh Onyxia service, before writing any model code:

    python scripts/check_env.py

It checks: GPU visibility, torch/CUDA versions, that all packages in
requirements.txt actually import, and whether the `mc` S3 client is aliased.
"""
import importlib
import subprocess
import sys


def check_packages() -> None:
    required = [
        "torch", "torchvision", "numpy", "matplotlib", "yaml",
        "tqdm", "tensorboard", "imageio", "boto3", "sklearn",
    ]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[FAIL] missing packages: {missing}")
        print("       -> pip install --user -r requirements.txt")
    else:
        print("[OK]   all required packages import successfully")


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError:
        print("[FAIL] torch is not importable at all — check the base image")
        return

    print(f"[INFO] torch version        : {torch.__version__}")
    print(f"[INFO] cuda available       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] gpu                  : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] cuda version (torch) : {torch.version.cuda}")
        # tiny real op on the GPU, not just a flag check
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"[OK]   ran a real matmul on GPU, result shape {tuple(y.shape)}")
    else:
        print("[FAIL] no GPU detected — check the resource request on your Onyxia service")


def check_s3() -> None:
    try:
        out = subprocess.run(
            ["mc", "alias", "list"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and out.stdout.strip():
            print("[OK]   mc aliases configured:")
            print(out.stdout)
        else:
            print("[WARN] mc found but no alias configured — see SETUP.md")
    except FileNotFoundError:
        print("[WARN] `mc` CLI not found on PATH — use boto3 fallback (see SETUP.md)")
    except subprocess.TimeoutExpired:
        print("[WARN] `mc alias list` timed out")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0 environment check")
    print("=" * 60)
    check_packages()
    print("-" * 60)
    check_torch_cuda()
    print("-" * 60)
    check_s3()
    print("=" * 60)
