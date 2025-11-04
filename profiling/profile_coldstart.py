#!/usr/bin/env python3
"""
Profile cold-start components: container init, model load, GPU load, KV allocation.
Outputs: profiling_outputs/cold_start.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

OUTPUT_DIR = Path(os.environ.get("PROFILE_OUTPUT_DIR", "profiling_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_run(model_id: str, device: str, kv_tokens: int) -> dict:
    t_process_start = time.time()
    container_init_s = time.time() - t_process_start

    t_model_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id)
    t_model_end = time.time()

    gpu_load_s = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        t_gpu_start = time.time()
        model = model.to(device)
        torch.cuda.synchronize()
        gpu_load_s = time.time() - t_gpu_start

    kv_alloc_s = 0.0
    if kv_tokens > 0 and device == "cuda":
        with torch.no_grad():
            input_ids = torch.randint(
                0,
                model.config.vocab_size,
                (1, kv_tokens),
                device=device,
                dtype=torch.long,
            )
            torch.cuda.synchronize()
            t_kv_start = time.time()
            _ = model(input_ids)
            torch.cuda.synchronize()
            kv_alloc_s = time.time() - t_kv_start

    total = container_init_s + (t_model_end - t_model_start) + gpu_load_s + kv_alloc_s
    return {
        "model": model_id,
        "device": device,
        "kv_tokens": kv_tokens,
        "container_init_s": container_init_s,
        "model_load_s": t_model_end - t_model_start,
        "gpu_load_s": gpu_load_s,
        "kv_alloc_s": kv_alloc_s,
        "total_cold_start_s": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("PROFILE_MODEL", "facebook/llama-7b"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--kv-tokens", type=int, default=512)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for run_id in range(args.runs):
        row = measure_run(args.model, args.device, args.kv_tokens)
        row["run_id"] = run_id
        rows.append(row)
        print(f"Run {run_id}: {row}")

    csv_path = OUTPUT_DIR / "cold_start.csv"
    fieldnames = [
        "run_id",
        "model",
        "device",
        "kv_tokens",
        "container_init_s",
        "model_load_s",
        "gpu_load_s",
        "kv_alloc_s",
        "total_cold_start_s",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
