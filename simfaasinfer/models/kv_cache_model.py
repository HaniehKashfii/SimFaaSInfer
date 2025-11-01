# simfaasinfer/models/kv_cache_model.py
"""
KV cache sizing and incremental allocation helpers.
"""
from typing import Dict, Any, List


def estimate_kv_size(request_meta: Dict[str, Any], model_config: Dict[str, Any]) -> int:
    """
    Return estimated KV cache size in bytes for a single request.
    request_meta: {"prefill_len": int, "decode_len": int,...}
    """
    num_layers = model_config.get('num_layers', 32)
    hidden_size = model_config.get('hidden_size', 4096)
    num_kv_heads = model_config.get('num_kv_heads', 32)
    head_dim = hidden_size // model_config.get('num_attention_heads', 32)
    
    # Total tokens for this request
    prefill_len = request_meta.get('prefill_len', 0)
    decode_len = request_meta.get('decode_len', 0)
    total_tokens = prefill_len + decode_len
    
    # KV cache per token: 2 (K and V) * num_layers * num_kv_heads * head_dim * bytes_per_element
    bytes_per_element = 4  # float32
    if model_config.get('kv_dtype') == 'float16':
        bytes_per_element = 2
    
    kv_size_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    total_kv_bytes = total_tokens * kv_size_per_token
    
    return total_kv_bytes


def estimated_incremental_alloc(request_list: List[Dict[str, Any]], page_size: int, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate incremental/paged allocation for a list of requests and return summary:
      {"total_kv_bytes":..., "alloc_events":[...]}
    """
    total_kv_bytes = 0
    alloc_events = []
    
    for req in request_list:
        req_kv_bytes = estimate_kv_size(req, model_config)
        total_kv_bytes += req_kv_bytes
        
        # Simulate paged allocation
        num_pages = (req_kv_bytes + page_size - 1) // page_size
        
        alloc_events.append({
            'request_id': req.get('request_id'),
            'kv_bytes': req_kv_bytes,
            'num_pages': num_pages,
            'timestamp': req.get('timestamp', 0)
        })
    
    return {
        'total_kv_bytes': total_kv_bytes,
        'alloc_events': alloc_events,
        'num_requests': len(request_list)
    }