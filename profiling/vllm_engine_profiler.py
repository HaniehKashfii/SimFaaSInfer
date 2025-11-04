#!/usr/bin/env python3
"""
Profile a vLLM engine using synthetic workloads and save latency statistics.

Outputs:
  - profiling_outputs/vllm_engine_profile.json     (TTFT/TBT per workload)
  - profiling_outputs/vllm_kernel_trace.json       (optional torch.profiler trace)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
from contextlib import nullcontext
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

OUTPUT_DIR = Path(os.environ.get("PROFILE_OUTPUT_DIR", "profiling_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 95),
    }


def make_prompt(tokenizer: AutoTokenizer, seq_len: int) -> str:
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    token_ids = [
        random.randint(10, vocab_size - 1)
        for _ in range(seq_len)
    ]
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)


def gather_request_metrics(outputs: List[RequestOutput]) -> Dict[str, List[float]]:
    ttft, tbt, total_latency, output_tokens = [], [], [], []
    for out in outputs:
        metrics = out.metrics
        if not metrics:
            continue
        seq_tokens = sum(len(comp.token_ids) for comp in out.outputs)
        arrival, first_token = metrics.arrival_time, metrics.first_token_time
        last_token, finished = metrics.last_token_time, metrics.finished_time
        if arrival is not None and first_token is not None:
            ttft.append(max(0.0, (first_token - arrival) * 1000.0))
        if first_token is not None and last_token is not None and seq_tokens:
            tbt_total = max(0.0, (last_token - first_token) * 1000.0)
            tbt.append(tbt_total / max(seq_tokens, 1))
        if arrival is not None and finished is not None:
            total_latency.append(max(0.0, (finished - arrival) * 1000.0))
        output_tokens.append(seq_tokens)
    return {
        "ttft_ms": ttft,
        "per_token_latency_ms": tbt,
        "end_to_end_ms": total_latency,
        "tokens": output_tokens,
    }


def run_workload(
    llm: LLM,
    tokenizer: AutoTokenizer,
    run_type: str,
    seq_len: int,
    batch_size: int,
    new_tokens: int,
    capture_trace: bool,
) -> Dict[str, object]:
    prompts = [make_prompt(tokenizer, seq_len) for _ in range(batch_size)]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=new_tokens,
        top_p=1.0,
    )
    profiler_ctx = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        )
        if capture_trace
        else nullcontext()
    )

    with profiler_ctx as prof:
        start = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        total_time = (time.time() - start) * 1000.0
        if capture_trace and prof:
            trace_path = OUTPUT_DIR / f"kernel_trace_{run_type}_seq{seq_len}_b{batch_size}.json"
            prof.export_chrome_trace(str(trace_path))

    metrics = gather_request_metrics(outputs)
    return {
        "run_type": run_type,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "max_new_tokens": new_tokens,
        "mean_output_tokens": statistics.mean(metrics["tokens"]) if metrics["tokens"] else 0,
        "ttft_ms": summarize(metrics["ttft_ms"]),
        "per_token_latency_ms": summarize(metrics["per_token_latency_ms"]),
        "end_to_end_ms": summarize(metrics["end_to_end_ms"]),
        "wall_time_ms": total_time,
    }


class nullcontext:
    """Simple context manager when torch.profiler is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile vLLM engine with synthetic workloads.")
    parser.add_argument("--model", default=os.environ.get("PROFILE_MODEL", "gpt2"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--prefill-seq", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--decode-kv", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--decode-new", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--capture-trace", action="store_true", help="Export torch.profiler trace for the last run.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens for prefill measurements.")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="auto")

    results = []
    for seq_len in args.prefill_seq:
        for bsz in args.batch_sizes:
            capture = args.capture_trace and not results  # capture trace on first workload if requested
            results.append(
                run_workload(llm, tokenizer, "prefill", seq_len, bsz, args.max_tokens, capture)
            )

    for kv in args.decode_kv:
        for new_tokens in args.decode_new:
            for bsz in args.batch_sizes[:2]:  # limit decode batches
                capture = args.capture_trace and False
                results.append(
                    run_workload(llm, tokenizer, "decode", kv, bsz, new_tokens, capture)
                )

    profile = {
        "model": args.model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "runs": results,
    }
    output_path = OUTPUT_DIR / "vllm_engine_profile.json"
    with output_path.open("w") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
