def estimate_resume_time(model_spec, tin, tout):
    """
    Estimate resume latency from migration-in/out durations.
    """
    resume_params = model_spec.get("migration", {}).get("resume_params", {})
    coef = float(resume_params.get("a_per_token_s", 0.0005))
    bias = float(resume_params.get("bias_s", 0.05))
    return coef * (float(tin) + float(tout)) + bias
