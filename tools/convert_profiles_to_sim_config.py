import argparse
import glob
import json
import os

import numpy as np
import yaml

from simfaasinfer.utils.io import validate_profile_artifact


def load_artifacts(dir_path):
    """Load and validate individual profile artifacts."""
    paths = glob.glob(os.path.join(dir_path, "*.json"))
    artifacts = []
    for path in paths:
        with open(path, "r") as f:
            artifact = json.load(f)
        try:
            validate_profile_artifact(artifact)
        except Exception:
            continue
        artifacts.append(artifact)
    return artifacts


def fit_attention_prefill(artifacts, num_layers):
    """Fit quadratic attention-prefill time per token."""
    rows = []
    ys = []
    for art in artifacts:
        if art.get("operator") != "attention_prefill":
            continue
        inputs = art.get("input", {})
        batch = inputs.get("batch_tokens", 1)
        seq = inputs.get("seq_len") or sum(inputs.get("seq_lens", []))
        seq = seq or 1
        y = art["runtime_ms"] / (max(1, batch) * max(1, num_layers))
        rows.append([seq * seq, seq, 1.0])
        ys.append(y)
    if not rows:
        return None
    X = np.array(rows)
    Y = np.array(ys)
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def fit_mlp(artifacts):
    """Median MLP time per token."""
    values = []
    for art in artifacts:
        if "mlp" in art.get("operator", ""):
            inputs = art.get("input", {})
            batch = inputs.get("batch_tokens", 1)
            values.append(art["runtime_ms"] / max(1, batch))
    if not values:
        return None
    return float(np.median(values))


def fit_comm(artifacts, op_name):
    """Median communication time normalized by GB."""
    values = []
    for art in artifacts:
        if art.get("operator") != op_name:
            continue
        msg = art.get("input", {}).get("msg_bytes", 0)
        if msg > 0:
            values.append(art["runtime_ms"] / (msg / 1e9))
    if not values:
        return None
    return float(np.median(values))


def main():
    parser = argparse.ArgumentParser(description="Convert profiling artifacts into a SimFaaSInfer model config.")
    parser.add_argument("--profiles_dir", required=True, help="Directory containing operator profile artifacts.")
    parser.add_argument("--model_desc", required=True, help="Base model descriptor YAML.")
    parser.add_argument("--loading_fit", required=True, help="JSON file from fit_loading_and_migration.py.")
    parser.add_argument("--out", required=True, help="Output path for generated SimFaaSInfer YAML.")
    args = parser.parse_args()

    artifacts = load_artifacts(args.profiles_dir)
    model = yaml.safe_load(open(args.model_desc, "r"))
    num_layers = model.get("num_layers", 32)

    attn_prefill = fit_attention_prefill(artifacts, num_layers)
    mlp_time = fit_mlp(artifacts)
    allreduce_time = fit_comm(artifacts, "allreduce")
    allgather_time = fit_comm(artifacts, "allgather")

    load_params = json.load(open(args.loading_fit, "r"))

    profiling = {}
    if attn_prefill:
        profiling["attention_prefill_time_per_token_sq"] = attn_prefill[0]
        profiling["attention_prefill_linear_term"] = attn_prefill[1]
    if mlp_time:
        profiling["mlp_time_per_token"] = mlp_time
    if allreduce_time:
        profiling["allreduce_time_per_gb"] = allreduce_time
    if allgather_time:
        profiling["allgather_time_per_gb"] = allgather_time
    profiling["model_memory_gb"] = model.get("model_size_gb", 0)
    profiling["kv_cache_memory_per_token_mb"] = model.get("kv_cache_memory_per_token_mb", 0.0005)

    cfg = {
        "model": {"name": model.get("model_id", "custom"), "model_id": model.get("model_id", "custom")},
        "profiling": profiling,
        "loading": {
            "storage_tier": "RAID0_NVMe",
            "storage_size_bytes": model.get("model_size_gb", 0) * 1e9,
            "loading_params": {
                "bandwidth_gb_s": load_params.get("bandwidth_gb_s", 12.0),
                "q_overhead_s": load_params.get("q_overhead_s", 0.0),
            },
        },
        "migration": {
            "resume_params": {
                "a_per_token_s": load_params.get("migration_resume_coef_a_per_token_s", 0.0),
                "bias_s": load_params.get("migration_resume_bias_s", 0.0),
            }
        },
        "cluster": {"num_replicas": 1, "gpu_type": "A100", "num_gpus_per_replica": 1},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
