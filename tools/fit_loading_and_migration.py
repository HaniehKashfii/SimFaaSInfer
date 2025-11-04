import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def fit_bandwidth(load_json_files):
    """Estimate median load bandwidth (GB/s) from profiling json files."""
    rows = []
    for path in load_json_files:
        with open(path, "r") as f:
            payload = json.load(f)
        if "duration_s" not in payload:
            continue
        size = payload.get("model_size_bytes", payload.get("size_bytes"))
        if size is None:
            continue
        duration = payload["duration_s"]
        rows.append((size / 1e9, duration))

    if not rows:
        return None

    ratios = [size_gb / duration for size_gb, duration in rows if duration > 0]
    if not ratios:
        return None
    return float(np.median(ratios))


def fit_migration(mig_json_files):
    """Fit resume time (s) as a*(tin+tout) + b."""
    X = []
    Y = []
    for path in mig_json_files:
        with open(path, "r") as f:
            payload = json.load(f)
        keys = ("tin", "tout", "resume_time_s")
        if not all(k in payload for k in keys):
            continue
        X.append([payload["tin"] + payload["tout"], 1.0])
        Y.append(payload["resume_time_s"])

    if not X:
        return None

    X_arr = np.array(X)
    Y_arr = np.array(Y)
    coeffs, *_ = np.linalg.lstsq(X_arr, Y_arr, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def main():
    parser = argparse.ArgumentParser(description="Fit loading bandwidth and migration resume parameters.")
    parser.add_argument("--load_dir", required=True, help="Directory containing loading profile JSONs.")
    parser.add_argument("--mig_dir", default=None, help="Directory containing migration profile JSONs.")
    parser.add_argument("--out", default="fitted_loading.json", help="Output JSON file.")
    args = parser.parse_args()

    load_files = glob.glob(os.path.join(args.load_dir, "*.json"))
    bandwidth = fit_bandwidth(load_files)

    result = {"bandwidth_gb_s": bandwidth}
    if args.mig_dir:
        mig_files = glob.glob(os.path.join(args.mig_dir, "*.json"))
        migration_coeffs = fit_migration(mig_files)
        if migration_coeffs:
            result["migration_resume_coef_a_per_token_s"] = migration_coeffs[0]
            result["migration_resume_bias_s"] = migration_coeffs[1]

    print("Fitted:", result)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
