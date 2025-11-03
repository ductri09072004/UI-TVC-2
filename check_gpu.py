#!/usr/bin/env python3
import subprocess
import shutil
import sys


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        return e.output or ""


def check_torch_cuda() -> None:
    try:
        import torch
    except Exception as e:
        print("PyTorch: not installed")
        return

    print(f"PyTorch: installed (version={torch.__version__})")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"  [{i}] {name} (capability={cap[0]}.{cap[1]})")
        try:
            current = torch.cuda.current_device()
            print(f"Current device index: {current}")
        except Exception:
            pass


def check_nvidia_smi() -> None:
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi: not found (NVIDIA driver/Toolkit may be missing)")
        return

    print("nvidia-smi: found")
    q_gpu = (
        "nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    out = run(q_gpu)
    if out.strip():
        print("\nGPU status:")
        for line in out.strip().splitlines():
            # index, name, driver, mem.total(MiB), mem.used(MiB), util(%)
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                idx, name, drv, mem_total, mem_used, util = parts[:6]
                print(f"  [{idx}] {name} | driver {drv} | mem {mem_used}/{mem_total} MiB | util {util}%")
            else:
                print("  "+line)

    q_procs = (
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    outp = run(q_procs)
    print("\nGPU processes:")
    if outp.strip():
        for line in outp.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_uuid, pid, name, mem = parts[:4]
                print(f"  pid={pid} mem={mem} MiB name={name} gpu={gpu_uuid}")
            else:
                print("  "+line)
    else:
        print("  (no active compute processes)")


def main() -> None:
    print("== GPU Check ==")
    check_torch_cuda()
    print("")
    check_nvidia_smi()


if __name__ == "__main__":
    main()


