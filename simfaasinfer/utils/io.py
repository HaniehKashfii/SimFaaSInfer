# simfaasinfer/utils/io.py
"""
IO helpers & JSON schema validation helpers for ProfileArtifact and TelemetrySample.
"""
import json
from typing import Dict, Any
from pathlib import Path

try:
    from jsonschema import validate, ValidationError
except ImportError:
    validate = None
    ValidationError = Exception


# JSON Schemas
PROFILE_ARTIFACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ProfileArtifact",
    "type": "object",
    "required": ["operator", "operator_class", "phase", "input", "tp", "pp", 
                 "runtime_ms", "gpu_memory_bytes", "timestamp"],
    "properties": {
        "operator": {"type": "string"},
        "operator_class": {"type": "string", "enum": ["token", "sequence", "comm"]},
        "phase": {"type": "string", "enum": ["prefill", "decode", "both"]},
        "input": {"type": "object", "additionalProperties": True},
        "tp": {"type": "integer", "minimum": 1},
        "pp": {"type": "integer", "minimum": 1},
        "runtime_ms": {"type": ["number", "null"]},
        "gpu_memory_bytes": {"type": ["integer", "null"]},
        "kernel_details": {"type": "object"},
        "timestamp": {"type": "number"}
    },
    "additionalProperties": True
}

TELEMETRY_SAMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TelemetrySample",
    "type": "object",
    "required": ["request_id", "phase", "batch_id", "batch_composition", 
                 "operator_timings", "gpu_mfu", "timestamp"],
    "properties": {
        "request_id": {"type": "string"},
        "phase": {"type": "string", "enum": ["prefill", "decode", "both"]},
        "batch_id": {"type": "string"},
        "batch_composition": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["request_id", "prefill_len", "kv_bytes"],
                "properties": {
                    "request_id": {"type": "string"},
                    "prefill_len": {"type": "integer"},
                    "decode_len": {"type": "integer"},
                    "kv_bytes": {"type": "integer"}
                }
            }
        },
        "operator_timings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "time_ms"],
                "properties": {
                    "name": {"type": "string"},
                    "time_ms": {"type": "number"}
                }
            }
        },
        "gpu_mfu": {"type": "number"},
        "timestamp": {"type": "number"},
        "extra": {"type": "object"}
    },
    "additionalProperties": True
}


def validate_profile_artifact(obj: Dict[str, Any]) -> bool:
    """Validate ProfileArtifact against schema."""
    if validate is None:
        print("Warning: jsonschema not installed, skipping validation")
        return True
    
    try:
        validate(obj, PROFILE_ARTIFACT_SCHEMA)
        return True
    except ValidationError as e:
        raise ValueError(f"ProfileArtifact validation error: {e.message}")


def validate_telemetry_sample(obj: Dict[str, Any]) -> bool:
    """Validate TelemetrySample against schema."""
    if validate is None:
        print("Warning: jsonschema not installed, skipping validation")
        return True
    
    try:
        validate(obj, TELEMETRY_SAMPLE_SCHEMA)
        return True
    except ValidationError as e:
        raise ValueError(f"TelemetrySample validation error: {e.message}")


def load_profile_artifacts(directory: str) -> list:
    """Load all profile artifacts from directory."""
    artifacts = []
    path = Path(directory)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for file_path in path.glob("*.json"):
        with open(file_path, 'r') as f:
            artifact = json.load(f)
            validate_profile_artifact(artifact)
            artifacts.append(artifact)
    
    return artifacts


def load_telemetry_samples(file_path: str) -> list:
    """Load telemetry samples from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        samples = data
    else:
        samples = data.get('samples', [])
    
    for sample in samples:
        validate_telemetry_sample(sample)
    
    return samples


def save_json(obj: Any, file_path: str, indent: int = 2):
    """Save object as JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)