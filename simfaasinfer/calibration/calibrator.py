# simfaasinfer/calibration/calibrator.py
"""
Calibrator fits residual/correction models that bias-correct RF estimator predictions using real telemetry.

Public:
    calibrate(estimator, telemetry_samples) -> calibrated_estimator
"""
from typing import List, Dict, Any
import numpy as np


class CalibratedEstimator:
    """Wrapper around estimator with calibration corrections."""
    
    def __init__(self, base_estimator, correction_models: Dict[str, Any]):
        self.base_estimator = base_estimator
        self.correction_models = correction_models
        self.confidence_scores = {}
    
    def predict(self, operator_query: Dict[str, Any]) -> float:
        """Predict with calibration correction."""
        # Get base prediction
        base_pred = self.base_estimator.predict(operator_query)
        
        # Apply correction
        op_class = operator_query.get('operator_class', 'token')
        if op_class in self.correction_models:
            correction = self.correction_models[op_class]
            # Simple multiplicative correction
            corrected_pred = base_pred * correction.get('scale_factor', 1.0) + correction.get('bias', 0.0)
            return max(0.01, corrected_pred)
        
        return base_pred
    
    def get_confidence(self, operator_query: Dict[str, Any]) -> float:
        """Return confidence score for prediction."""
        op_class = operator_query.get('operator_class', 'token')
        return self.confidence_scores.get(op_class, 0.5)


def calibrate(estimator: Any, telemetry_samples: List[Dict[str, Any]]) -> CalibratedEstimator:
    """
    Fit correction mapping on estimator predictions vs telemetry residuals.

    Args:
        estimator: Base RFEstimator
        telemetry_samples: List of TelemetrySample dicts with actual runtimes

    Returns:
        CalibratedEstimator with correction models
    """
    print(f"Calibrating estimator with {len(telemetry_samples)} telemetry samples...")
    
    # Group telemetry by operator class
    operator_telemetry = {}
    for sample in telemetry_samples:
        operator_timings = sample.get('operator_timings', [])
        
        for op_timing in operator_timings:
            op_name = op_timing.get('name', 'unknown')
            actual_time = op_timing.get('time_ms', 0)
            
            # Infer operator class from name
            op_class = _infer_operator_class(op_name)
            
            if op_class not in operator_telemetry:
                operator_telemetry[op_class] = {'predictions': [], 'actuals': []}
            
            # Get prediction from base estimator
            query = _construct_query_from_telemetry(sample, op_name, op_class)
            try:
                predicted_time = estimator.predict(query)
                operator_telemetry[op_class]['predictions'].append(predicted_time)
                operator_telemetry[op_class]['actuals'].append(actual_time)
            except:
                continue
    
    # Fit correction models
    correction_models = {}
    confidence_scores = {}
    
    for op_class, data in operator_telemetry.items():
        predictions = np.array(data['predictions'])
        actuals = np.array(data['actuals'])
        
        if len(predictions) < 5:
            print(f"  {op_class}: Insufficient data ({len(predictions)} samples), skipping")
            continue
        
        # Compute residuals
        residuals = actuals - predictions
        relative_errors = (actuals - predictions) / actuals
        
        # Fit simple linear correction: actual = scale * predicted + bias
        scale_factor = np.mean(actuals) / np.mean(predictions) if np.mean(predictions) > 0 else 1.0
        bias = np.mean(residuals)
        
        # Confidence score (based on R²)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        correction_models[op_class] = {
            'scale_factor': scale_factor,
            'bias': bias,
            'mean_relative_error': np.mean(np.abs(relative_errors)),
            'r_squared': r_squared
        }
        
        confidence_scores[op_class] = max(0, r_squared)
        
        print(f"  {op_class}: scale={scale_factor:.3f}, bias={bias:.3f}ms, "
              f"R²={r_squared:.3f}, samples={len(predictions)}")
    
    calibrated = CalibratedEstimator(estimator, correction_models)
    calibrated.confidence_scores = confidence_scores
    
    print("Calibration complete")
    return calibrated


def _infer_operator_class(op_name: str) -> str:
    """Infer operator class from operator name."""
    name_lower = op_name.lower()
    if 'attention' in name_lower or 'attn' in name_lower:
        return 'sequence'
    elif 'mlp' in name_lower or 'linear' in name_lower:
        return 'token'
    elif 'reduce' in name_lower or 'gather' in name_lower:
        return 'comm'
    else:
        return 'token'


def _construct_query_from_telemetry(sample: Dict[str, Any], op_name: str, op_class: str) -> Dict[str, Any]:
    """Construct operator query from telemetry sample."""
    batch_composition = sample.get('batch_composition', [])
    
    query = {
        'operator_name': op_name,
        'operator_class': op_class,
        'phase': sample.get('phase', 'both'),
        'inputs': {},
        'tp': 1,
        'pp': 1,
        'model_spec': {},
        'hardware': {}
    }
    
    # Extract inputs from batch composition
    if op_class == 'token':
        total_tokens = sum([req.get('prefill_len', 0) + req.get('decode_len', 0) 
                           for req in batch_composition])
        query['inputs'] = {'batch_tokens': total_tokens}
    
    elif op_class == 'sequence':
        if sample.get('phase') == 'prefill':
            seq_lens = [req.get('prefill_len', 512) for req in batch_composition]
            query['inputs'] = {'seq_lens': seq_lens}
        else:
            total_kv = sum([req.get('kv_bytes', 0) for req in batch_composition])
            query['inputs'] = {
                'total_kv_bytes': total_kv,
                'batch_size': len(batch_composition)
            }
    
    return query