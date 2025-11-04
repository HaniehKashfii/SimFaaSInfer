"""Model configuration and specifications."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for an LLM model.

    Contains architectural details and profiled performance characteristics.
    """

    # Model identification
    name: str
    architecture: str

    # Architecture parameters
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    max_position_embeddings: int

    # Attention configuration
    attention_type: str  # 'mha' or 'gqa'
    head_dim: int
    kv_channels: int

    # Profiling data (time in milliseconds)
    attention_prefill_time_per_token_sq: float
    attention_decode_time_per_token: float
    mlp_time_per_token: float
    layernorm_time_per_token: float
    embedding_time_per_token: float

    # Communication operators (ms per GB)
    allreduce_time_per_gb: float
    allgather_time_per_gb: float
    send_recv_time_per_gb: float

    # Memory footprint
    model_memory_gb: float
    kv_cache_memory_per_token_mb: float

    def __init__(self, config: Dict):
        """Initialize from configuration dictionary.

        Args:
            config: Configuration dictionary
        """
        model_config = config
        profiling = config.get('profiling', {})

        self.name = model_config['name']
        self.architecture = model_config.get('architecture', 'llama')

        # Architecture
        self.num_layers = model_config['num_layers']
        self.hidden_size = model_config['hidden_size']
        self.intermediate_size = model_config['intermediate_size']
        self.num_attention_heads = model_config['num_attention_heads']
        self.num_kv_heads = model_config['num_kv_heads']
        self.vocab_size = model_config['vocab_size']
        self.max_position_embeddings = model_config['max_position_embeddings']

        # Attention
        self.attention_type = model_config.get('attention_type', 'mha')
        self.head_dim = model_config.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.kv_channels = model_config.get('kv_channels', self.num_kv_heads * self.head_dim)

        # Profiling
        self.attention_prefill_time_per_token_sq = profiling.get('attention_prefill_time_per_token_sq', 0.000015)
        self.attention_decode_time_per_token = profiling.get('attention_decode_time_per_token', 0.0002)
        self.mlp_time_per_token = profiling.get('mlp_time_per_token', 0.0003)
        self.layernorm_time_per_token = profiling.get('layernorm_time_per_token', 0.00003)
        self.embedding_time_per_token = profiling.get('embedding_time_per_token', 0.00004)

        # Communication
        self.allreduce_time_per_gb = profiling.get('allreduce_time_per_gb', 5.0)
        self.allgather_time_per_gb = profiling.get('allgather_time_per_gb', 4.5)
        self.send_recv_time_per_gb = profiling.get('send_recv_time_per_gb', 3.0)

        # Memory
        self.model_memory_gb = profiling.get('model_memory_gb', 14.0)
        self.kv_cache_memory_per_token_mb = profiling.get('kv_cache_memory_per_token_mb', 0.0005)

        predictors_cfg = profiling.get('predictors', {})
        self.prefill_predictor_path: Optional[str] = predictors_cfg.get('prefill')
        self.decode_predictor_path: Optional[str] = predictors_cfg.get('decode')

    def get_kv_cache_size_per_token(self) -> float:
        """Get KV-cache size per token in MB.

        Returns:
            KV-cache size in MB for one token across all layers
        """
        return self.kv_cache_memory_per_token_mb * self.num_layers

    def get_total_kv_cache_size(self, num_tokens: int) -> float:
        """Get total KV-cache size in MB.

        Args:
            num_tokens: Number of tokens

        Returns:
            Total KV-cache size in MB
        """
        return self.get_kv_cache_size_per_token() * num_tokens

    def supports_gqa(self) -> bool:
        """Check if model uses Grouped Query Attention.

        Returns:
            True if model uses GQA
        """
        return self.attention_type == 'gqa' and self.num_kv_heads < self.num_attention_heads

    def __repr__(self) -> str:
        """String representation."""
        return (f"ModelConfig(name={self.name}, layers={self.num_layers}, "
                f"hidden={self.hidden_size}, attention={self.attention_type})")
