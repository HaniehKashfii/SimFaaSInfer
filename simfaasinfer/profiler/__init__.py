# simfaasinfer/profiler/__init__.py
"""Profiling module for operator runtime characterization."""

from .operator_triage import generate_profiling_plan, ProfilingPlan
from .profiler_runner import run_profiles
from .parallelism_adapter import device_operator_mapping
from .ingest_cupti import parse_cupti_trace

__all__ = [
    "generate_profiling_plan",
    "ProfilingPlan",
    "run_profiles",
    "device_operator_mapping",
    "parse_cupti_trace",
]