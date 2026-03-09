#!/usr/bin/env python3
"""Resource monitor for GPU and CPU usage. No dependencies beyond stdlib."""

import subprocess
import time
import sys
from collections import deque


def get_gpu_stats():
    """Get GPU utilization and memory from nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                gpus.append(
                    {
                        "util": int(parts[0]),
                        "mem_used": int(parts[1]),
                        "mem_total": int(parts[2]),
                    }
                )
            return gpus
    except Exception:
        pass
    return None


def get_cpu_stats():
    """Get CPU utilization from /proc/stat (Linux only)."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        # user, nice, system, idle, iowait, irq, softirq, steal
        idle = int(parts[4])
        total = sum(int(p) for p in parts[1:])
        return idle, total
    except Exception:
        return None, None


def get_mem_stats():
    """Get memory usage from /proc/meminfo (Linux only)."""
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                mem[key] = int(parts[1])  # Value in kB
        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", 0)
        used = total - available
        return {"used": used // 1024, "total": total // 1024}  # Convert to MB
    except Exception:
        return None


def calc_cpu_percent(prev_idle, prev_total, curr_idle, curr_total):
    """Calculate CPU percentage between two samples."""
    idle_delta = curr_idle - prev_idle
    total_delta = curr_total - prev_total
    if total_delta == 0:
        return 0.0
    return 100.0 * (1.0 - idle_delta / total_delta)


def format_bar(percent, width=20):
    """Create a simple ASCII progress bar."""
    filled = int(width * percent / 100)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    history_size = 30  # Keep last N samples for stats

    gpu_history = deque(maxlen=history_size)
    cpu_history = deque(maxlen=history_size)
    mem_history = deque(maxlen=history_size)

    prev_idle, prev_total = get_cpu_stats()

    print(f"Monitoring resources (interval={interval}s, Ctrl+C to stop)\n")
    print("-" * 60)

    try:
        while True:
            time.sleep(interval)

            # CPU
            curr_idle, curr_total = get_cpu_stats()
            cpu_pct = None
            if prev_idle is not None and curr_idle is not None:
                cpu_pct = calc_cpu_percent(prev_idle, prev_total, curr_idle, curr_total)
                cpu_history.append(cpu_pct)
                prev_idle, prev_total = curr_idle, curr_total

            # Memory
            mem = get_mem_stats()
            mem_pct = None
            if mem:
                mem_pct = 100 * mem["used"] / mem["total"]
                mem_history.append(mem_pct)

            # GPU
            gpus = get_gpu_stats()
            if gpus:
                gpu_history.append(gpus[0]["util"])

            # Clear line and print stats
            timestamp = time.strftime("%H:%M:%S")
            lines = [f"\033[2K{timestamp}"]

            if cpu_pct is not None:
                cpu_avg = sum(cpu_history) / len(cpu_history) if cpu_history else 0
                lines.append(
                    f"  CPU: {format_bar(cpu_pct)} {cpu_pct:5.1f}%  "
                    f"(avg: {cpu_avg:5.1f}%, min: {min(cpu_history):5.1f}%, max: {max(cpu_history):5.1f}%)"
                )

            if mem and mem_pct is not None:
                mem_avg = sum(mem_history) / len(mem_history) if mem_history else 0
                lines.append(
                    f"  RAM: {format_bar(mem_pct)} {mem['used']:5}MB / {mem['total']}MB  "
                    f"(avg: {mem_avg:5.1f}%, max: {max(mem_history):5.1f}%)"
                )

            if gpus:
                for i, gpu in enumerate(gpus):
                    gpu_avg = sum(gpu_history) / len(gpu_history) if gpu_history else 0
                    mem_pct = 100 * gpu["mem_used"] / gpu["mem_total"]
                    lines.append(
                        f"  GPU {i}: {format_bar(gpu['util'])} {gpu['util']:5.1f}%  "
                        f"(avg: {gpu_avg:5.1f}%, min: {min(gpu_history):5.1f}%, max: {max(gpu_history):5.1f}%)"
                    )
                    lines.append(
                        f"  Mem {i}: {format_bar(mem_pct)} {gpu['mem_used']:5}MB / {gpu['mem_total']}MB"
                    )

            print("\n".join(lines))
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n\nFinal Summary:")
        print("=" * 60)
        if cpu_history:
            print(
                f"  CPU:  avg={sum(cpu_history)/len(cpu_history):.1f}%  "
                f"min={min(cpu_history):.1f}%  max={max(cpu_history):.1f}%  "
                f"samples={len(cpu_history)}"
            )
        if mem_history:
            print(
                f"  RAM:  avg={sum(mem_history)/len(mem_history):.1f}%  "
                f"min={min(mem_history):.1f}%  max={max(mem_history):.1f}%  "
                f"samples={len(mem_history)}"
            )
        if gpu_history:
            print(
                f"  GPU:  avg={sum(gpu_history)/len(gpu_history):.1f}%  "
                f"min={min(gpu_history):.1f}%  max={max(gpu_history):.1f}%  "
                f"samples={len(gpu_history)}"
            )
        print()


if __name__ == "__main__":
    main()
