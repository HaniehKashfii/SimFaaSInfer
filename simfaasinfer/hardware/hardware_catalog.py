# simfaasinfer/hardware/hardware_catalog.py
"""
Hardware catalog for GPU SKUs & instance definitions.
YAML-driven loader returning GPU specs and pricing.
"""
from typing import Dict, Any
import yaml
import os


class HardwareCatalog:
    def __init__(self, yaml_path: str = None):
        self.catalog = {}
        if yaml_path and os.path.exists(yaml_path):
            self.load_catalog(yaml_path)
        else:
            self._load_default_catalog()

    def load_catalog(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.catalog = yaml.safe_load(f)

    def _load_default_catalog(self):
        """Load default hardware catalog."""
        self.catalog = {
            'gpus': {
                'A100': {
                    'memory_gb': 80,
                    'tflops_fp32': 19.5,
                    'tflops_fp16': 312,
                    'mem_bw_gb_s': 2000,
                    'nvlink_bw_gb_s': 600,
                    'pcie_bw_gb_s': 32,
                    'price_per_hour': 3.06
                },
                'H100': {
                    'memory_gb': 80,
                    'tflops_fp32': 51,
                    'tflops_fp16': 989,
                    'mem_bw_gb_s': 3350,
                    'nvlink_bw_gb_s': 900,
                    'pcie_bw_gb_s': 128,
                    'price_per_hour': 5.12
                },
                'A10G': {
                    'memory_gb': 24,
                    'tflops_fp32': 31.2,
                    'tflops_fp16': 125,
                    'mem_bw_gb_s': 600,
                    'nvlink_bw_gb_s': 0,  # No NVLink
                    'pcie_bw_gb_s': 32,
                    'price_per_hour': 1.21
                },
            },
            'instances': {
                'p4d.24xlarge': {
                    'gpu_type': 'A100',
                    'num_gpus': 8,
                    'topology': 'nvlink',
                    'price_per_hour': 32.77
                },
                'p5.48xlarge': {
                    'gpu_type': 'H100',
                    'num_gpus': 8,
                    'topology': 'nvlink',
                    'price_per_hour': 98.32
                },
                'g5.48xlarge': {
                    'gpu_type': 'A10G',
                    'num_gpus': 8,
                    'topology': 'pcie',
                    'price_per_hour': 16.29
                },
            }
        }

    def get_gpu_sku(self, name: str) -> Dict[str, Any]:
        return self.catalog.get("gpus", {}).get(name, {})

    def list_instances(self):
        return list(self.catalog.get("instances", {}).keys())
    
    def get_instance_spec(self, name: str) -> Dict[str, Any]:
        return self.catalog.get("instances", {}).get(name, {})