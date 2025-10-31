"""Execution time prediction for LLM operators."""

import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor

from .model_config import ModelConfig
from ..utils.logger import setup_logger


class ExecutionTimePredictor:
    """Predict execution time for LLM operators.

    Uses profiled data and machine learning models to predict runtime
    for various batch compositions and parallelization strategies.
    """

    def __init__(self, model_config: ModelConfig, parallelization_config: Dict):
        """Initialize predictor.

        Args:
            model_config: Model configuration
            parallelization_config: Parallelization settings (TP, PP, replicas)
        """
        self.model_config = model_config
        self.parallelization = parallelization_config
        self.logger = setup_logger(self.__class__.__name__)

        self.tensor_parallel_size = parallelization_config.get('tensor_parallel_size', 1)
        self.pipeline_parallel_size = parallelization_config.get('pipeline_parallel_size', 1)

        # ML models for runtime prediction (optional)
        self.prefill_predictor = None
        self.decode_predictor = None

    def predict_prefill_time(self, num_tokens: int, batch_size: int) -> float:
        """Predict prefill phase execution time.

        Args:
            num_tokens: Total number of tokens in prefill
            batch_size: Number of sequences

        Returns:
            Predicted execution time in milliseconds
        """
        # Attention: O(n^2) complexity for prefill
        # Approximate as single sequence with equivalent compute
        effective_length = int(np.sqrt(num_tokens))
        attention_time = (
                self.model_config.attention_prefill_time_per_token_sq *
                effective_length ** 2 *
                self.model_config.num_layers
        )

        # MLP: Linear in number of tokens
        mlp_time = (
                self.model_config.mlp_time_per_token *
                num_tokens *
                self.model_config.num_layers
        )

        # LayerNorm
        layernorm_time = (
                self.model_config.layernorm_time_per_token *
                num_tokens *
                self.model_config.num_layers * 2  # 2 per layer
        )

        # Embedding
        embedding_time = self.model_config.embedding_time_per_token * num_tokens

        # Total compute time
        compute_time = attention_time + mlp_time + layernorm_time + embedding_time

        # Adjust for tensor parallelism
        compute_time /= self.tensor_parallel_size

        # Add communication overhead for tensor parallelism
        if self.tensor_parallel_size > 1:
            # All-reduce after each layer
            data_size_gb = (num_tokens * self.model_config.hidden_size * 4) / (1024 ** 3)  # float32
            comm_time = (
                    self.model_config.allreduce_time_per_gb *
                    data_size_gb *
                    self.model_config.num_layers
            )
            compute_time += comm_time

        # Pipeline parallelism: divide by stages
        compute_time /= self.pipeline_parallel_size

        return compute_time

    def predict_decode_time(self, num_decode_tokens: int, total_kv_tokens: int,
                            batch_size: int) -> float:
        """Predict decode phase execution time.

        Args:
            num_decode_tokens: Number of new tokens to generate
            total_kv_tokens: Total KV-cache size across batch
            batch_size: Number of sequences

        Returns:
            Predicted execution time in milliseconds
        """
        # Attention: Memory-bound, depends on KV-cache size
        attention_time = (
                self.model_config.attention_decode_time_per_token *
                total_kv_tokens *
                self.model_config.num_layers
        )

        # MLP: Linear in decode tokens
        mlp_time = (
                self.model_config.mlp_time_per_token *
                num_decode_tokens *
                self.model_config.num_layers
        )

        # LayerNorm
        layernorm_time = (
                self.model_config.layernorm_time_per_token *
                num_decode_tokens *
                self.model_config.num_layers * 2
        )

        # Embedding
        embedding_time = self.model_config.embedding_time_per_token * num_decode_tokens

        # Total compute time
        compute_time = attention_time + mlp_time + layernorm_time + embedding_time

        # Adjust for tensor parallelism
        compute_time /= self.tensor_parallel_size

        # Add communication overhead
        if self.tensor_parallel_size > 1:
            data_size_gb = (num_decode_tokens * self.model_config.hidden_size * 4) / (1024 ** 3)
            comm_time = (
                    self.model_config.allreduce_time_per_gb *
                    data_size_gb *
                    self.model_config.num_layers
            )
            compute_time += comm_time

        # Pipeline parallelism
        compute_time /= self.pipeline_parallel_size

        return compute_time

    def predict_batch_time(self, prefill_tokens: int, decode_tokens: int,
                           kv_cache_tokens: int, batch_size: int) -> float:
        """Predict total batch execution time.

        Args:
            prefill_tokens: Total prefill tokens in batch
            decode_tokens: Total decode tokens in batch
            kv_cache_tokens: Total KV-cache size
            batch_size: Number of sequences

        Returns:
            Predicted execution time in milliseconds
        """
        total_time = 0.0

        if prefill_tokens > 0:
            total_time += self.predict_prefill_time(prefill_tokens, batch_size)

        if decode_tokens > 0:
            total_time += self.predict_decode_time(decode_tokens, kv_cache_tokens, batch_size)

        return total_time

    def train_predictor(self, training_data: List[Dict]) -> None:
        """Train ML models from profiled data (optional enhancement).

        Args:
            training_data: List of profiling samples with features and runtimes
        """
        if not training_data:
            return

        # Extract features and labels for prefill
        prefill_data = [d for d in training_data if d['type'] == 'prefill']
        if prefill_data:
            X_prefill = np.array([[d['num_tokens'], d['batch_size']] for d in prefill_data])
            y_prefill = np.array([d['runtime'] for d in prefill_data])

            self.prefill_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.prefill_predictor.fit(X_prefill, y_prefill)
            self.logger.info("Trained prefill predictor")

        # Extract features and labels for decode
        decode_data = [d for d in training_data if d['type'] == 'decode']
        if decode_data:
            X_decode = np.array([
                [d['num_tokens'], d['kv_cache_tokens'], d['batch_size']]
                for d in decode_data
            ])
            y_decode = np.array([d['runtime'] for d in decode_data])

            self.decode_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.decode_predictor.fit(X_decode, y_decode)
            self.logger.info("Trained decode predictor")