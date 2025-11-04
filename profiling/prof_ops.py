#!/usr/bin/env python3
"""Operator, memory, and communication profiling for SimFaaSInfer (Vidur layout)."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("PROFILE_MODEL", "facebook/llama-7b")
LOCAL_FILES_ONLY = os.getenv("PROFILE_LOCAL_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
BASE_OUTPUT = Path(os.environ.get("PROFILE_OUTPUT_DIR", "profiling_outputs"))

ATTENTION_DIR = BASE_OUTPUT / "attention"
MLP_DIR = BASE_OUTPUT / "mlp"
COLLECTIVES_DIR = BASE_OUTPUT / "collectives"
MEMORY_DIR = BASE_OUTPUT / "memory"
ENGINE_DIR = BASE_OUTPUT / "engine"
TRACE_DIR = ATTENTION_DIR / "traces"

for directory in [
    BASE_OUTPUT,
    ATTENTION_DIR,
    MLP_DIR,
    COLLECTIVES_DIR,
    MEMORY_DIR,
    ENGINE_DIR,
    TRACE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL, use_fast=False, local_files_only=LOCAL_FILES_ONLY
)
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=LOCAL_FILES_ONLY).to(DEVICE)
model.eval()

MAX_SEQ_LEN = getattr(model.config, "max_position_embeddings", 2048)
NUM_LAYERS = getattr(model.config, "num_hidden_layers", 1)
HIDDEN_SIZE = getattr(model.config, "hidden_size", 4096)
TP_DEGREE = int(os.getenv("PROFILE_TP_DEGREE", "1"))
PP_DEGREE = int(os.getenv("PROFILE_PP_DEGREE", "1"))

ATTENTION_COLUMNS = [
    "model_name",
    "run_id",
    "type",
    "seq_len",
    "batch_size",
    "kv_cache",
    "tp_degree",
    "pp_degree",
    "num_layers",
    "hidden_size",
    "time_mean_ms",
    "time_median_ms",
    "time_std_ms",
    "kernel_breakdown_json",
]

MLP_COLUMNS = [
    "model_name",
    "run_id",
    "type",
    "seq_len",
    "batch_size",
    "tp_degree",
    "pp_degree",
    "num_layers",
    "hidden_size",
    "time_mean_ms",
    "time_median_ms",
    "time_std_ms",
    "kernel_breakdown_json",
]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_region(fn, *args, repeats: int = 8, warmups: int = 3):
    for _ in range(warmups):
        with torch.no_grad():
            _ = fn(*args)
    times: List[float] = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fn(*args)
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return np.mean(times), np.median(times), np.std(times)


def run_operator_profiler(
    run_type: str,
    seq_len: int,
    batch_size: int,
    *,
    kv_cache: Optional[int] = None,
    new_tokens: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Gather kernel-level breakdown for a representative workload."""
    if run_type == "prefill":
        input_tensor = torch.randint(
            0, tokenizer.vocab_size, (batch_size, seq_len), device=DEVICE, dtype=torch.long
        )

        def forward():
            return model(input_tensor)

    elif run_type == "decode":
        if kv_cache is None or new_tokens is None:
            raise ValueError("decode profiler requires kv_cache and new_tokens")
        prefix = torch.randint(
            0, tokenizer.vocab_size, (batch_size, kv_cache), device=DEVICE, dtype=torch.long
        )
        tail = torch.randint(
            0, tokenizer.vocab_size, (batch_size, new_tokens), device=DEVICE, dtype=torch.long
        )
        concat = torch.cat([prefix, tail], dim=1)

        def forward():
            return model(concat)

    else:
        raise ValueError(f"Unsupported run_type {run_type}")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        with record_function("inference_forward"):
            with torch.no_grad():
                _ = forward()

    trace_name = TRACE_DIR / f"trace_{run_type}_seq{seq_len}_b{batch_size}.json"
    prof.export_chrome_trace(str(trace_name))

    breakdown_rows: List[Dict[str, float]] = []
    for evt in prof.key_averages():
        cuda_total = getattr(evt, "cuda_time_total", getattr(evt, "device_time_total", 0.0)) or 0.0
        cuda_mean = getattr(evt, "cuda_time_mean", getattr(evt, "device_time", 0.0)) or 0.0
        cpu_total = getattr(evt, "cpu_time_total", 0.0) or 0.0
        breakdown_rows.append(
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
            }
        )

    breakdown_csv = ATTENTION_DIR / "operator_breakdown.csv"
    write_mode = "w" if not breakdown_csv.exists() else "a"
    with breakdown_csv.open(write_mode, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_type",
                "seq_len",
                "batch_size",
                "op_name",
                "cuda_time_total_ms",
                "cuda_time_mean_ms",
                "cpu_time_total_ms",
                "cuda_occurrences",
                "input_shapes",
            ],
        )
        if write_mode == "w":
            writer.writeheader()
        writer.writerows(breakdown_rows)

    top_ops = sorted(breakdown_rows, key=lambda r: r["cuda_time_total_ms"], reverse=True)[:15]
    summary = [
        {
            "op": row["op_name"],
            "cuda_total_ms": round(row["cuda_time_total_ms"], 4),
            "count": int(row["cuda_occurrences"]),
        }
        for row in top_ops
    ]
    return summary


def profile_prefill(seq_len: int, batch_size: int):
    if seq_len > MAX_SEQ_LEN:
        return None
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, seq_len), device=DEVICE, dtype=torch.long
    )
    mean, median, std = time_region(lambda x: model(x), input_ids)
    return {"mean": mean, "median": median, "std": std}


def profile_decode(kv_cache: int, new_tokens: int, batch_size: int):
    if kv_cache + new_tokens > MAX_SEQ_LEN:
        return None
    prefix = torch.randint(
        0, tokenizer.vocab_size, (batch_size, kv_cache), device=DEVICE, dtype=torch.long
    )
    tail = torch.randint(
        0, tokenizer.vocab_size, (batch_size, new_tokens), device=DEVICE, dtype=torch.long
    )

    def forward():
        return model(torch.cat([prefix, tail], dim=1))

    mean, median, std = time_region(forward)
    return {"mean": mean, "median": median, "std": std}


def resolve_mlp_module():
    transformer_block = getattr(model, "transformer", None)
    if transformer_block and hasattr(transformer_block, "h"):
        block = transformer_block.h[0]
        if hasattr(block, "mlp"):
            return block.mlp
    raise RuntimeError("Unable to locate MLP module on this model; extend resolve_mlp_module().")


MLP_MODULE = resolve_mlp_module()


def profile_mlp(seq_len: int, batch_size: int):
    dtype = next(MLP_MODULE.parameters()).dtype
    hidden = torch.randn(batch_size, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=dtype)

    def forward(x):
        return MLP_MODULE(x)

    mean, median, std = time_region(forward, hidden)
    return {"mean": mean, "median": median, "std": std}


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_model_memory() -> Dict[str, float]:
    total_param_bytes = sum(p.element_size() * p.nelement() for p in model.parameters())
    total_buffer_bytes = sum(b.element_size() * b.nelement() for b in model.buffers())
    total_bytes = total_param_bytes + total_buffer_bytes
    info = {
        "model_name": MODEL,
        "param_bytes": total_param_bytes,
        "buffer_bytes": total_buffer_bytes,
        "total_bytes": total_bytes,
        "model_size_gb": total_bytes / (1024**3),
        "dtype": str(next(model.parameters()).dtype),
        "num_parameters": sum(p.nelement() for p in model.parameters()),
    }
    with (MEMORY_DIR / "model_memory.json").open("w") as f:
        json.dump(info, f, indent=2)
    return info


def profile_kv_cache_memory(batch_size: int = 2, kv_points: Optional[Sequence[int]] = None):
    if not torch.cuda.is_available():
        print("CUDA not available; skipping KV cache memory profiling.")
        return []
    kv_points = kv_points or [64, 128, 256, 512, 1024]
    rows = []
    for kv in kv_points:
        if kv > MAX_SEQ_LEN:
            continue
        torch.cuda.empty_cache()
        _sync()
        before = torch.cuda.memory_reserved()
        prefix = torch.randint(
            0, tokenizer.vocab_size, (batch_size, kv), device=DEVICE, dtype=torch.long
        )
        with torch.no_grad():
            _ = model(prefix)
        _sync()
        after = torch.cuda.memory_reserved()
        delta_mb = max(0.0, (after - before) / (1024**2))
        rows.append(
            {
                "kv_cache": kv,
                "batch_size": batch_size,
                "mem_mb": delta_mb,
            }
        )
        del prefix
        torch.cuda.empty_cache()
    write_csv(MEMORY_DIR / "kv_cache.csv", ["kv_cache", "batch_size", "mem_mb"], rows)
    return rows


def measure_pcie_bandwidth(size_bytes_list=None, repeats: int = 3):
    if not torch.cuda.is_available():
        return []
    size_bytes_list = size_bytes_list or [
        int(0.1 * 1024**3),
        int(0.5 * 1024**3),
        int(1 * 1024**3),
    ]
    rows = []
    for size_bytes in size_bytes_list:
        num_elements = size_bytes // 4
        timings = []
        for _ in range(repeats):
            host_tensor = torch.randn(num_elements, device="cpu", dtype=torch.float32)
            _sync()
            t0 = time.perf_counter()
            _ = host_tensor.to(DEVICE)
            _sync()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
            del host_tensor
        median_ms = statistics.median(timings)
        bandwidth = (size_bytes / (1024**3)) / (median_ms / 1000.0) if median_ms > 0 else 0.0
        rows.append(
            {
                "collective_type": "host_to_device_copy",
                "size_bytes": size_bytes,
                "time_ms": median_ms,
                "bandwidth_gbps": bandwidth,
            }
        )
    return rows


def write_collective_csvs(rows: List[Dict[str, float]]):
    if not rows:
        return
    write_csv(
        COLLECTIVES_DIR / "allreduce.csv",
        ["collective_type", "size_bytes", "time_ms", "bandwidth_gbps"],
        rows,
    )
    write_csv(
        COLLECTIVES_DIR / "send_recv.csv",
        ["collective_type", "size_bytes", "time_ms", "bandwidth_gbps"],
        rows,
    )
    ms_per_gb = [1000.0 / r["bandwidth_gbps"] for r in rows if r["bandwidth_gbps"] > 0]
    avg_ms = statistics.mean(ms_per_gb) if ms_per_gb else 0.0
    comm_estimates = {
        "assumption": "Single-GPU PCIe host->device copy approximates collective costs.",
        "allreduce_time_per_gb": avg_ms,
        "allgather_time_per_gb": avg_ms,
        "send_recv_time_per_gb": avg_ms,
    }
    with (COLLECTIVES_DIR / "communication_estimates.json").open("w") as f:
        json.dump(comm_estimates, f, indent=2)
    return comm_estimates


def build_attention_rows(prefill_summary, decode_summary):
    rows = []
    run_id = 0
    kernel_prefill = json.dumps({"top_ops": prefill_summary})
    kernel_decode = json.dumps({"top_ops": decode_summary})

    for seq in [64, 128, 256, 512, 1024]:
        for batch in [1, 2, 4, 8]:
            stats = profile_prefill(seq, batch)
            if not stats:
                continue
            rows.append(
                {
                    "model_name": MODEL,
                    "run_id": run_id,
                    "type": "prefill",
                    "seq_len": seq,
                    "batch_size": batch,
                    "kv_cache": 0,
                    "tp_degree": TP_DEGREE,
                    "pp_degree": PP_DEGREE,
                    "num_layers": NUM_LAYERS,
                    "hidden_size": HIDDEN_SIZE,
                    "time_mean_ms": stats["mean"],
                    "time_median_ms": stats["median"],
                    "time_std_ms": stats["std"],
                    "kernel_breakdown_json": kernel_prefill,
                }
            )
            run_id += 1

    for kv in [128, 512]:
        for new_tokens in [4, 16, 64]:
            for batch in [1, 2]:
                stats = profile_decode(kv, new_tokens, batch)
                if not stats:
                    continue
                rows.append(
                    {
                        "model_name": MODEL,
                        "run_id": run_id,
                        "type": "decode",
                        "seq_len": new_tokens,
                        "batch_size": batch,
                        "kv_cache": kv,
                        "tp_degree": TP_DEGREE,
                        "pp_degree": PP_DEGREE,
                        "num_layers": NUM_LAYERS,
                        "hidden_size": HIDDEN_SIZE,
                        "time_mean_ms": stats["mean"],
                        "time_median_ms": stats["median"],
                        "time_std_ms": stats["std"],
                        "kernel_breakdown_json": kernel_decode,
                    }
                )
                run_id += 1
    write_csv(ATTENTION_DIR / "attention.csv", ATTENTION_COLUMNS, rows)
    return rows


def build_mlp_rows(prefill_summary):
    rows = []
    run_id = 0
    kernel_json = json.dumps({"top_ops": prefill_summary})
    for seq in [32, 64, 128, 256]:
        for batch in [1, 2, 4]:
            stats = profile_mlp(seq, batch)
            rows.append(
                {
                    "model_name": MODEL,
                    "run_id": run_id,
                    "type": "prefill",
                    "seq_len": seq,
                    "batch_size": batch,
                    "tp_degree": TP_DEGREE,
                    "pp_degree": PP_DEGREE,
                    "num_layers": 1,
                    "hidden_size": HIDDEN_SIZE,
                    "time_mean_ms": stats["mean"],
                    "time_median_ms": stats["median"],
                    "time_std_ms": stats["std"],
                    "kernel_breakdown_json": kernel_json,
                }
            )
            run_id += 1
    write_csv(MLP_DIR / "mlp.csv", MLP_COLUMNS, rows)
    return rows


def compute_linear_slope(points: Sequence[Dict[str, float]]) -> float:
    if not points:
        return 0.0
    xs = [p["kv_cache"] * p["batch_size"] for p in points]
    ys = [p["mem_mb"] for p in points]
    n = len(points)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-6:
        return 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope


def compute_operator_constants(attention_rows, mlp_rows, breakdown_rows):
    def mean_or_zero(values):
        return statistics.mean(values) if values else 0.0

    prefill_consts = []
    for row in attention_rows:
        if row["type"] != "prefill":
            continue
        denom = max(1, (row["seq_len"] ** 2) * NUM_LAYERS)
        prefill_consts.append((row["time_mean_ms"] / 1000.0) / denom)

    decode_consts = []
    for row in attention_rows:
        if row["type"] != "decode" or row["kv_cache"] <= 0:
            continue
        denom = max(1, row["kv_cache"] * NUM_LAYERS)
        decode_consts.append((row["time_mean_ms"] / 1000.0) / denom)

    mlp_consts = [
        (row["time_mean_ms"] / 1000.0) / max(1, row["seq_len"] * row["batch_size"])
        for row in mlp_rows
    ]

    layernorm_ms = []
    embedding_ms = []
    for row in breakdown_rows:
        op = row["op_name"]
        if "layer_norm" in op:
            layernorm_ms.append(row["cuda_time_total_ms"] / 1000.0)
        if "embedding" in op:
            embedding_ms.append(row["cuda_time_total_ms"] / 1000.0)

    return {
        "attention_prefill": mean_or_zero(prefill_consts),
        "attention_decode": mean_or_zero(decode_consts),
        "mlp": mean_or_zero(mlp_consts),
        "layernorm": mean_or_zero(layernorm_ms),
        "embedding": mean_or_zero(embedding_ms),
    }


def load_breakdown_rows() -> List[Dict[str, float]]:
    csv_path = ATTENTION_DIR / "operator_breakdown.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return [
            {
                "op_name": row["op_name"],
                "cuda_time_total_ms": float(row["cuda_time_total_ms"]),
            }
            for row in reader
        ]


def write_fitted_profile(
    attention_rows,
    mlp_rows,
    memory_info,
    comm_estimates,
    kv_rows,
):
    breakdown_rows = load_breakdown_rows()
    operators = compute_operator_constants(attention_rows, mlp_rows, breakdown_rows)

    slope_mb = compute_linear_slope(kv_rows)
    kv_per_token_mb = slope_mb

    profile = {
        "model_name": MODEL,
        "operators": operators,
        "communication": {
            "allreduce_time_per_gb": comm_estimates.get("allreduce_time_per_gb", 0.0),
            "allgather_time_per_gb": comm_estimates.get("allgather_time_per_gb", 0.0),
            "send_recv_time_per_gb": comm_estimates.get("send_recv_time_per_gb", 0.0),
        },
        "memory": {
            "model_size_gb": memory_info.get("model_size_gb", 0.0),
            "kv_cache_per_token_mb": kv_per_token_mb,
        },
    }
    fitted_path = BASE_OUTPUT / "fitted_profile.json"
    with fitted_path.open("w") as f:
        json.dump(profile, f, indent=2)

    safe_model = MODEL.replace("/", "_")
    sim_profile_dir = Path("SimFaaSInfer/data/profiling/compute/a100") / safe_model
    sim_profile_dir.mkdir(parents=True, exist_ok=True)
    with (sim_profile_dir / "fitted_profile.json").open("w") as f:
        json.dump(profile, f, indent=2)


def main():
    print("Running torch.profiler reference runs...")
    attention_summary = run_operator_profiler("prefill", seq_len=512, batch_size=8)
    decode_summary = run_operator_profiler("decode", seq_len=64, batch_size=2, kv_cache=512, new_tokens=32)

    print("Collecting attention samples...")
    attention_rows = build_attention_rows(attention_summary, decode_summary)

    print("Collecting MLP samples...")
    mlp_rows = build_mlp_rows(attention_summary)

    print("Recording memory/KV cache metrics...")
    memory_info = record_model_memory()
    kv_rows = profile_kv_cache_memory()

    print("Measuring PCIe/collective bandwidth...")
    comm_rows = measure_pcie_bandwidth()
    comm_estimates = write_collective_csvs(comm_rows)

    print("Building fitted_profile.json...")
    write_fitted_profile(attention_rows, mlp_rows, memory_info, comm_estimates or {}, kv_rows)

    print("Profiling complete. Outputs stored under profiling_outputs/.")


if __name__ == "__main__":
    main()
