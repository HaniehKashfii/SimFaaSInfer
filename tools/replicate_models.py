import argparse
import json
import os
from pathlib import Path

import yaml


def replicate_model(src_yaml, out_dir, base_id, count, size_bytes=None):
    """
    Duplicate an existing model descriptor while tweaking model_id (and optionally size).
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(src_yaml, "r") as f:
        desc = yaml.safe_load(f)

    models = []
    for i in range(count):
        replica_id = f"{base_id}-replica-{i:03d}"
        desc_copy = dict(desc)
        desc_copy["model_id"] = replica_id
        if size_bytes is not None:
            loading = dict(desc_copy.get("loading") or {})
            loading["storage_size_bytes"] = int(size_bytes)
            desc_copy["loading"] = loading

        out_path = Path(out_dir) / f"{replica_id}.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(desc_copy, f, sort_keys=False)
        models.append(str(out_path))

    return models


def main():
    parser = argparse.ArgumentParser(description="Replicate model descriptors for placement experiments.")
    parser.add_argument("--src", required=True, help="Path to source model YAML descriptor.")
    parser.add_argument(
        "--out_dir",
        default="SimFaaSInfer/configs/models/replicas",
        help="Directory to place replicated descriptors.",
    )
    parser.add_argument("--base", required=True, help="Base identifier used for replica names.")
    parser.add_argument("--count", type=int, required=True, help="Number of replicas to generate.")
    parser.add_argument("--size_bytes", type=int, default=None, help="Optional override for storage_size_bytes.")
    args = parser.parse_args()

    replicas = replicate_model(args.src, args.out_dir, args.base, args.count, args.size_bytes)
    manifest = {"replicas": replicas}
    manifest_path = Path(args.out_dir) / "replicas_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(replicas)} replicas to {args.out_dir}")


if __name__ == "__main__":
    main()
