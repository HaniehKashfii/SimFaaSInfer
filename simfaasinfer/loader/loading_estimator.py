def estimate_loading_time(model_spec, server_state):
    """
    Estimate loading time by combining queue delay and transfer latency.
    """
    loading = model_spec.get("loading", {})
    params = loading.get("loading_params", {})
    bandwidth_gb_s = float(params.get("bandwidth_gb_s", 12.0))
    storage_size_bytes = float(loading.get("storage_size_bytes", model_spec.get("model_size_gb", 0) * 1e9))
    queue_delay = float(server_state.get("loading_queue_estimated_delay", 0.0))

    size_gb = storage_size_bytes / 1e9
    transfer_time = size_gb / bandwidth_gb_s if bandwidth_gb_s > 0 else float("inf")
    return queue_delay + transfer_time
