"""
Simple profile-based lookup estimator.
Uses fitted profile data for runtime estimation.
"""
from typing import Dict, Any, List
import numpy as np


class ProfileLookupEstimator:
    """
    Simple estimator that looks up runtimes from profiling data.
    Used as baseline when RF estimators are not available.
    """

    def __init__(self, profile_data: List[Dict[str, Any]]):
        """
        Initialize with profile data.

        Args:
            profile_data: List of profile entries from fitted_profile.json
        """
        self.profile_data = profile_data

        # Index by operation type
        self.prefill_data = [p for p in profile_data if p.get('type') == 'prefill']
        self.decode_data = [p for p in profile_data if p.get('type') == 'decode']

        # Build lookup tables
        self.prefill_lookup = {}
        self.decode_lookup = {}

        for entry in self.prefill_data:
            key = (entry.get('seq_len'), entry.get('batch_size'))
            self.prefill_lookup[key] = entry.get('mean_ms', 0)

        for entry in self.decode_data:
            key = (entry.get('kv_cache'), entry.get('new_tokens'), entry.get('batch_size'))
            self.decode_lookup[key] = entry.get('mean_ms', 0)

    def predict(self, operator_query: Dict[str, Any]) -> float:
        """
        Predict runtime for operator.

        Args:
            operator_query: Query dict with operation parameters

        Returns:
            Estimated runtime in milliseconds
        """
        op_class = operator_query.get('operator_class', 'token')
        phase = operator_query.get('phase', 'both')

        # Determine operation type
        if 'prefill' in phase.lower():
            return self._predict_prefill(operator_query)
        elif 'decode' in phase.lower():
            return self._predict_decode(operator_query)
        else:
            # Average of both
            return (self._predict_prefill(operator_query) +
                   self._predict_decode(operator_query)) / 2

    def _predict_prefill(self, query: Dict[str, Any]) -> float:
        """Predict prefill runtime."""
        inputs = query.get('inputs', {})

        # Extract parameters
        seq_len = inputs.get('seq_len', 512)
        batch_size = inputs.get('batch_size', 1)

        # Handle seq_lens list
        if isinstance(inputs.get('seq_lens'), list):
            seq_len = int(np.mean(inputs['seq_lens']))
            batch_size = len(inputs['seq_lens'])

        # Try exact lookup
        key = (seq_len, batch_size)
        if key in self.prefill_lookup:
            return self.prefill_lookup[key]

        # Interpolate/extrapolate
        return self._interpolate_prefill(seq_len, batch_size)

    def _predict_decode(self, query: Dict[str, Any]) -> float:
        """Predict decode runtime."""
        inputs = query.get('inputs', {})

        # Extract parameters
        kv_cache = inputs.get('kv_cache', 512)
        new_tokens = inputs.get('new_tokens', 1)
        batch_size = inputs.get('batch_size', 1)

        # Try exact lookup
        key = (kv_cache, new_tokens, batch_size)
        if key in self.decode_lookup:
            return self.decode_lookup[key]

        # Interpolate/extrapolate
        return self._interpolate_decode(kv_cache, new_tokens, batch_size)

    def _interpolate_prefill(self, seq_len: int, batch_size: int) -> float:
        """Interpolate prefill time."""
        if not self.prefill_lookup:
            # Fallback: rough estimate
            return seq_len * batch_size * 0.02  # ~0.02ms per token

        # Find closest entries
        closest_entries = []
        for (s, b), time_ms in self.prefill_lookup.items():
            distance = abs(s - seq_len) + abs(b - batch_size)
            closest_entries.append((distance, time_ms, s, b))

        closest_entries.sort()

        # Use weighted average of 3 closest
        if len(closest_entries) >= 3:
            weights = [1.0, 0.5, 0.25]
            weighted_sum = sum(w * e[1] for w, e in zip(weights, closest_entries[:3]))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        elif len(closest_entries) > 0:
            return closest_entries[0][1]
        else:
            return seq_len * batch_size * 0.02

    def _interpolate_decode(self, kv_cache: int, new_tokens: int, batch_size: int) -> float:
        """Interpolate decode time."""
        if not self.decode_lookup:
            # Fallback: rough estimate
            return kv_cache * batch_size * 0.01  # ~0.01ms per cached token

        # Find closest entries
        closest_entries = []
        for (k, n, b), time_ms in self.decode_lookup.items():
            distance = abs(k - kv_cache) + abs(n - new_tokens) + abs(b - batch_size)
            closest_entries.append((distance, time_ms, k, n, b))

        closest_entries.sort()

        # Use weighted average of 3 closest
        if len(closest_entries) >= 3:
            weights = [1.0, 0.5, 0.25]
            weighted_sum = sum(w * e[1] for w, e in zip(weights, closest_entries[:3]))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        elif len(closest_entries) > 0:
            return closest_entries[0][1]
        else:
            return kv_cache * batch_size * 0.01

    def get_capabilities(self) -> Dict[str, Any]:
        """Return estimator capabilities."""
        return {
            'type': 'ProfileLookup',
            'num_prefill_entries': len(self.prefill_lookup),
            'num_decode_entries': len(self.decode_lookup),
            'prefill_seq_lens': sorted(set(k[0] for k in self.prefill_lookup.keys())),
            'prefill_batch_sizes': sorted(set(k[1] for k in self.prefill_lookup.keys())),
            'decode_kv_caches': sorted(set(k[0] for k in self.decode_lookup.keys()))
        }
