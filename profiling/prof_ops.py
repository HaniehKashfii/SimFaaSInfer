#!/usr/bin/env python3
import csv
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL = os.environ.get("PROFILE_MODEL", "facebook/llama-7b")
OUTPUT_DIR = Path(os.environ.get("PROFILE_OUTPUT_DIR", "profiling_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
model.eval()
MAX_SEQ_LEN = getattr(model.config, "max_position_embeddings", 2048)


def time_region(fn, *args, repeats=10, warmups=5):
    for _ in range(warmups):
        with torch.no_grad():
            _ = fn(*args)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = fn(*args)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return np.mean(times), np.median(times), np.std(times)


def profile_prefill(seq_len, batch_size):
    if seq_len > MAX_SEQ_LEN:
        return None
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, seq_len), device=DEVICE, dtype=torch.long
    )
    mean, med, std = time_region(lambda x: model(x), input_ids)
    return {
        "type": "prefill",
        "seq_len": seq_len,
        "batch_size": batch_size,
        "mean_ms": mean,
        "median_ms": med,
        "std_ms": std,
    }


def profile_decode(kv_cache, new_tokens, batch_size):
    if kv_cache + new_tokens > MAX_SEQ_LEN:
        return None
    prefix = torch.randint(
        0, tokenizer.vocab_size, (batch_size, kv_cache), device=DEVICE, dtype=torch.long
    )
    new_input = torch.randint(
        0,
        tokenizer.vocab_size,
        (batch_size, new_tokens),
        device=DEVICE,
        dtype=torch.long,
    )

    def forward_combined():
        _ = model(torch.cat([prefix, new_input], dim=1))

    mean, med, std = time_region(forward_combined, repeats=8)
    return {
        "type": "decode",
        "kv_cache": kv_cache,
        "new_tokens": new_tokens,
        "batch_size": batch_size,
        "mean_ms": mean,
        "median_ms": med,
        "std_ms": std,
    }


def append_operator_rows(rows):
    if not rows:
        return
    csv_path = OUTPUT_DIR / "operator_breakdown.csv"
    fieldnames = [
        "run_type",
        "seq_len",
        "batch_size",
        "op_name",
        "cuda_time_total_ms",
        "cuda_time_mean_ms",
        "cpu_time_total_ms",
        "cuda_occurrences",
        "input_shapes",
        "extra_tags",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_operator_profiler(run_type, seq_len, batch_size, kv_cache=None, new_tokens=None):
    if run_type == "prefill":
        input_tensor = torch.randint(
            0, tokenizer.vocab_size, (batch_size, seq_len), device=DEVICE, dtype=torch.long
        )

        def forward():
            return model(input_tensor)

    elif run_type == "decode":
        if kv_cache is None or new_tokens is None:
            raise ValueError("decode profiler needs kv_cache and new_tokens")
        if kv_cache + new_tokens > MAX_SEQ_LEN:
            return
        prefix = torch.randint(
            0, tokenizer.vocab_size, (batch_size, kv_cache), device=DEVICE, dtype=torch.long
        )
        new_input = torch.randint(
            0,
            tokenizer.vocab_size,
            (batch_size, new_tokens),
            device=DEVICE,
            dtype=torch.long,
        )
        concat = torch.cat([prefix, new_input], dim=1)

        def forward():
            return model(concat)

    else:
        raise ValueError(f"Unknown run_type {run_type}")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        with record_function("inference_forward"):
            with torch.no_grad():
                _ = forward()

    trace_name = f"trace_{run_type}_seq{seq_len}_b{batch_size}.json"
    prof.export_chrome_trace(str(OUTPUT_DIR / trace_name))

    rows = []
    for evt in prof.key_averages():
        cuda_total = getattr(evt, "cuda_time_total", getattr(evt, "device_time_total", 0.0)) or 0.0
        cuda_mean = getattr(evt, "cuda_time_mean", getattr(evt, "device_time", 0.0)) or 0.0
        cpu_total = getattr(evt, "cpu_time_total", 0.0) or 0.0
        rows.append(
            {
                "run_type": run_type,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "op_name": evt.key,
                "cuda_time_total_ms": cuda_total / 1000.0,
                "cuda_time_mean_ms": cuda_mean / 1000.0,
                "cpu_time_total_ms": cpu_total / 1000.0,
                "cuda_occurrences": evt.count,
                "input_shapes": str(getattr(evt, "input_shapes", "")),
                "extra_tags": str(getattr(evt, "extra_fields", "")),
            }
        )
    append_operator_rows(rows)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    print(table)


def record_model_memory():
    total_param_bytes = sum(p.element_size() * p.nelement() for p in model.parameters())
    total_buffer_bytes = sum(b.element_size() * b.nelement() for b in model.buffers())
    total_bytes = total_param_bytes + total_buffer_bytes
    return {
        "model_name": MODEL,
        "param_bytes": total_param_bytes,
        "buffer_bytes": total_buffer_bytes,
        "total_bytes": total_bytes,
        "model_size_gb": total_bytes / (1024**3),
        "dtype": str(next(model.parameters()).dtype),
        "num_parameters": sum(p.nelement() for p in model.parameters()),
    }


def profile_kv_cache_memory(batch_size=2, kv_points=None):
    if not torch.cuda.is_available():
        print("CUDA not available; skipping KV cache memory profiling.")
        return []
    kv_points = kv_points or [64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    for kv in kv_points:
        if kv > MAX_SEQ_LEN:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        before = torch.cuda.memory_reserved()
        prefix = torch.randint(
            0, tokenizer.vocab_size, (batch_size, kv), device=DEVICE, dtype=torch.long
        )
        with torch.no_grad():
            _ = model(prefix)
        torch.cuda.synchronize()
        after = torch.cuda.memory_reserved()
        delta_mb = max(0.0, (after - before) / (1024**2))
        rows.append({"kv_cache": kv, "batch_size": batch_size, "mem_mb": delta_mb})
        del prefix
        torch.cuda.empty_cache()
    return rows


def write_kv_cache_rows(rows):
    if not rows:
        return
    csv_path = OUTPUT_DIR / "kv_cache_memory.csv"
    fieldnames = ["kv_cache", "batch_size", "mem_mb"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def measure_pcie_bandwidth(size_bytes_list=None, repeats=3):
    if not torch.cuda.is_available():
        print("CUDA not available; skipping PCIe bandwidth test.")
        return []
    size_bytes_list = size_bytes_list or [
        int(0.1 * 1024**3),
        int(0.5 * 1024**3),
        int(1 * 1024**3),
    ]
    results = []
    for size_bytes in size_bytes_list:
        num_elements = size_bytes // 4
        times_ms = []
        for _ in range(repeats):
            x = torch.randn(num_elements, device="cpu", dtype=torch.float32)
            torch.cuda.synchronize()
            t0 = time.time()
            _ = x.to(DEVICE)
            torch.cuda.synchronize()
            t1 = time.time()
            times_ms.append((t1 - t0) * 1000.0)
            del x
        median_ms = statistics.median(times_ms)
        bandwidth_gbps = (size_bytes / (1024**3)) / (median_ms / 1000.0) if median_ms > 0 else 0.0
        results.append(
            {
                "collective_type": "host_to_device_copy",
                "size_bytes": size_bytes,
                "time_ms": median_ms,
                "bandwidth_gbps": bandwidth_gbps,
            }
        )
    return results


def write_comm_csv(rows):
    if not rows:
        return
    csv_path = OUTPUT_DIR / "comm_bandwidth.csv"
    fieldnames = ["collective_type", "size_bytes", "time_ms", "bandwidth_gbps"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_comm_estimates(rows):
    if not rows:
        return
    ms_per_gb = [1000.0 / r["bandwidth_gbps"] for r in rows if r["bandwidth_gbps"] > 0]
    avg_ms = statistics.mean(ms_per_gb) if ms_per_gb else None
    estimates = {
        "assumption": "Single-GPU PCIe host->device transfer used as proxy; replace with NCCL data when available.",
        "allreduce_time_per_gb_ms": avg_ms,
        "allgather_time_per_gb_ms": avg_ms,
        "send_recv_time_per_gb_ms": avg_ms,
        "source": "host_to_device_copy",
    }
    with (OUTPUT_DIR / "communication_estimates.json").open("w") as f:
        json.dump(estimates, f, indent=2)


if __name__ == "__main__":
    samples = []
    for seq in [64, 128, 256, 512, 1024]:
        for b in [1, 2, 4, 8]:
            if seq > MAX_SEQ_LEN:
                continue
            print("Prefill", seq, b)
            result = profile_prefill(seq, b)
            if result:
                samples.append(result)
    for kv in [128, 512, 2048]:
        for new in [1, 8, 32]:
            for b in [1, 2]:
                if kv + new > MAX_SEQ_LEN:
                    continue
                print("Decode", kv, new, b)
                result = profile_decode(kv, new, b)
                if result:
                    samples.append(result)

    with (OUTPUT_DIR / "operators.json").open("w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {(OUTPUT_DIR / 'operators.json')}")

    memory_info = record_model_memory()
    with (OUTPUT_DIR / "model_memory.json").open("w") as f:
        json.dump(memory_info, f, indent=2)
    print(f"Saved {(OUTPUT_DIR / 'model_memory.json')}")

    kv_rows = profile_kv_cache_memory(batch_size=2)
    write_kv_cache_rows(kv_rows)
    if kv_rows:
        print(f"Saved {(OUTPUT_DIR / 'kv_cache_memory.csv')}")

    comm_rows = measure_pcie_bandwidth()
    write_comm_csv(comm_rows)
    write_comm_estimates(comm_rows)
    if comm_rows:
        print(f"Saved {(OUTPUT_DIR / 'comm_bandwidth.csv')} and communication_estimates.json")

    print("Running torch.profiler for operator breakdown (prefill seq512 batch8)...")
    run_operator_profiler("prefill", seq_len=min(512, MAX_SEQ_LEN), batch_size=8)
