"""Load and process workload traces."""

import json
import csv
from pathlib import Path
from typing import List, Dict

from ..utils.logger import setup_logger


class TraceLoader:
    """Load workload traces from various formats.

    Supports:
    - JSON format
    - CSV format
    - Custom formats (LMSys, Arxiv, etc.)
    """

    def __init__(self, trace_path: str):
        """Initialize trace loader.

        Args:
            trace_path: Path to trace file
        """
        self.trace_path = Path(trace_path)
        self.logger = setup_logger(self.__class__.__name__)

        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

    def load(self) -> List[Dict]:
        """Load trace data.

        Returns:
            List of trace entries with timestamp, prompt_tokens, decode_tokens
        """
        suffix = self.trace_path.suffix.lower()

        if suffix == '.json':
            return self._load_json()
        elif suffix == '.csv':
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported trace format: {suffix}")

    def _load_json(self) -> List[Dict]:
        """Load JSON trace.

        Returns:
            List of trace entries
        """
        with open(self.trace_path, 'r') as f:
            data = json.load(f)

        # Normalize format
        trace = []
        for entry in data:
            trace.append({
                'timestamp': entry.get('timestamp', 0),
                'prompt_tokens': entry.get('prompt_tokens', entry.get('num_prompt_tokens', 512)),
                'decode_tokens': entry.get('decode_tokens', entry.get('num_decode_tokens', 128)),
            })

        self.logger.info(f"Loaded {len(trace)} entries from JSON trace")
        return trace

    def _load_csv(self) -> List[Dict]:
        """Load CSV trace.

        Returns:
            List of trace entries
        """
        trace = []

        with open(self.trace_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace.append({
                    'timestamp': float(row.get('timestamp', 0)),
                    'prompt_tokens': int(row.get('prompt_tokens', row.get('num_prompt_tokens', 512))),
                    'decode_tokens': int(row.get('decode_tokens', row.get('num_decode_tokens', 128))),
                })

        self.logger.info(f"Loaded {len(trace)} entries from CSV trace")
        return trace

    @staticmethod
    def generate_sample_trace(output_path: str, num_entries: int = 1000,
                              trace_type: str = 'chat') -> None:
        """Generate a sample trace file for testing.

        Args:
            output_path: Output file path
            num_entries: Number of entries to generate
            trace_type: Type of trace (chat, arxiv, bwb)
        """
        import numpy as np

        if trace_type == 'chat':
            # ChatGPT-like: short prompts, short responses
            prompt_mean, prompt_std = 400, 200
            decode_mean, decode_std = 150, 80
        elif trace_type == 'arxiv':
            # Summarization: long prompts, medium responses
            prompt_mean, prompt_std = 3000, 500
            decode_mean, decode_std = 300, 100
        elif trace_type == 'bwb':
            # Translation: medium prompts, long responses
            prompt_mean, prompt_std = 1500, 400
            decode_mean, decode_std = 2000, 600
        else:
            prompt_mean, prompt_std = 512, 256
            decode_mean, decode_std = 128, 64

        trace = []
        current_time = 0.0

        for i in range(num_entries):
            # Poisson arrivals (rate = 10 QPS)
            inter_arrival = np.random.exponential(0.1)
            current_time += inter_arrival

            # Sample lengths
            prompt_tokens = int(np.clip(np.random.gamma(
                (prompt_mean / prompt_std) ** 2,
                prompt_std ** 2 / prompt_mean
            ), 10, 4096))

            decode_tokens = int(np.clip(np.random.gamma(
                (decode_mean / decode_std) ** 2,
                decode_std ** 2 / decode_mean
            ), 1, 2048))

            trace.append({
                'timestamp': current_time,
                'prompt_tokens': prompt_tokens,
                'decode_tokens': decode_tokens,
            })

        # Save as JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(trace, f, indent=2)

        print(f"Generated sample trace: {output_path} ({num_entries} entries)")