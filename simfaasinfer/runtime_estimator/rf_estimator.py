# simfaasinfer/runtime_estimator/rf_estimator.py
"""
RandomForest-based runtime estimator.

Public class:
    RFEstimator
"""
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import os
import json

# scikit-learn import guarded for the skeleton - dev should ensure dependency installed
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except Exception:
    RandomForestRegressor = None

from .estimator_utils import (
    features_for_token_op,
    features_for_sequence_op,
    features_for_comm_op,
    normalize_features
)


class RFEstimator:
    def __init__(self):
        """Initialize container for per-operator RandomForest models."""
        self.models = {}  # operator_name -> trained model
        self.feature_info = {}  # metadata for features used
        self.fitted = False

    def fit(self, profile_artifacts: List[Dict[str, Any]], model_spec: Dict[str, Any] = None, hw_spec: Dict[str, Any] = None):
        """
        Fit per-operator RandomForest models using profile artifacts.

        Args:
            profile_artifacts: list of ProfileArtifact dicts.
            model_spec: model specification dict
            hw_spec: hardware specification dict
        """
        if RandomForestRegressor is None:
            raise RuntimeError("scikit-learn required to run RFEstimator.fit")
        
        if not profile_artifacts:
            raise ValueError("No profile artifacts provided for training")
        
        # Use defaults if not provided
        if model_spec is None:
            model_spec = {'hidden_size': 4096, 'num_layers': 32}
        if hw_spec is None:
            hw_spec = {'mem_bw_gb_s': 2000, 'tflops': 312, 'nvlink_bw_gb_s': 600}
        
        # Group artifacts by operator class
        operator_data = {}
        for artifact in profile_artifacts:
            op_class = artifact.get('operator_class', 'token')
            if op_class not in operator_data:
                operator_data[op_class] = []
            operator_data[op_class].append(artifact)
        
        print(f"Training RF models on {len(profile_artifacts)} artifacts...")
        
        # Train model for each operator class
        for op_class, artifacts in operator_data.items():
            X_list = []
            y_list = []
            
            for artifact in artifacts:
                # Extract features based on operator class
                if op_class == 'token':
                    features = features_for_token_op(
                        model_spec,
                        artifact['input'],
                        hw_spec
                    )
                elif op_class == 'sequence':
                    features = features_for_sequence_op(
                        model_spec,
                        artifact['input'],
                        hw_spec,
                        phase=artifact.get('phase', 'prefill')
                    )
                elif op_class == 'comm':
                    features = features_for_comm_op(
                        artifact['input'],
                        hw_spec
                    )
                else:
                    continue
                
                # Add parallelism features
                features['tp'] = artifact.get('tp', 1)
                features['pp'] = artifact.get('pp', 1)
                
                # Normalize
                features = normalize_features(features)
                
                # Convert to array
                feature_vec = [features[k] for k in sorted(features.keys())]
                X_list.append(feature_vec)
                y_list.append(artifact['runtime_ms'])
            
            if len(X_list) < 5:
                print(f"Warning: Only {len(X_list)} samples for {op_class}, skipping training")
                continue
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Train RandomForest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            
            # Store model and feature names
            self.models[op_class] = model
            self.feature_info[op_class] = sorted(features.keys())
            
            # Compute training score
            train_score = model.score(X, y)
            print(f"  {op_class}: R² = {train_score:.4f}, samples = {len(X)}")
        
        self.fitted = True
        print(f"Training complete. Models for: {list(self.models.keys())}")

    def predict(self, operator_query: Dict[str, Any]) -> float:
        """
        Predict runtime (ms) for a single operator query.

        operator_query keys: operator_name, operator_class, phase, inputs (dict), tp, pp, 
                             model_spec (dict), hardware (dict)
        """
        if not self.fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        
        op_class = operator_query.get('operator_class', 'token')
        
        if op_class not in self.models:
            # Fallback to simple heuristic
            return self._fallback_predict(operator_query)
        
        # Extract features
        model_spec = operator_query.get('model_spec', {})
        hw_spec = operator_query.get('hardware', {})
        inputs = operator_query.get('inputs', {})
        phase = operator_query.get('phase', 'both')
        
        if op_class == 'token':
            features = features_for_token_op(model_spec, inputs, hw_spec)
        elif op_class == 'sequence':
            features = features_for_sequence_op(model_spec, inputs, hw_spec, phase)
        elif op_class == 'comm':
            features = features_for_comm_op(inputs, hw_spec)
        else:
            return self._fallback_predict(operator_query)
        
        # Add parallelism
        features['tp'] = operator_query.get('tp', 1)
        features['pp'] = operator_query.get('pp', 1)
        
        # Normalize
        features = normalize_features(features)
        
        # Convert to array matching training feature order
        feature_names = self.feature_info[op_class]
        feature_vec = np.array([features.get(k, 0.0) for k in feature_names])
        
        # Predict
        prediction = self.models[op_class].predict(feature_vec.reshape(1, -1))[0]
        
        return max(0.01, float(prediction))

    def _fallback_predict(self, operator_query: Dict[str, Any]) -> float:
        """Fallback deterministic prediction when no model available."""
        op_class = operator_query.get('operator_class', 'token')
        inputs = operator_query.get('inputs', {})
        
        if op_class == 'token':
            batch_tokens = inputs.get('batch_tokens', 128)
            return batch_tokens * 0.0003
        elif op_class == 'sequence':
            if operator_query.get('phase') == 'prefill':
                seq_len = inputs.get('seq_len', 512)
                return seq_len * seq_len * 0.000015
            else:
                total_kv_bytes = inputs.get('total_kv_bytes', 1024 * 1024)
                return total_kv_bytes / (2e9) * 1000  # 2 GB/s effective BW
        elif op_class == 'comm':
            msg_bytes = inputs.get('msg_bytes', 1024 * 1024)
            return msg_bytes / (600e9) * 1000
        
        return 1.0

    def save(self, path: str):
        """Persist estimator metadata and models to `path` directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        model_path = os.path.join(path, 'models.joblib')
        joblib.dump(self.models, model_path)
        
        # Save metadata
        metadata = {
            'feature_info': self.feature_info,
            'fitted': self.fitted,
            'model_classes': list(self.models.keys())
        }
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved estimator to {path}")

    def load(self, path: str):
        """Load estimator from `path` created by save()."""
        model_path = os.path.join(path, 'models.joblib')
        metadata_path = os.path.join(path, 'metadata.json')
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Estimator files not found in {path}")
        
        self.models = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_info = metadata['feature_info']
        self.fitted = metadata['fitted']
        
        print(f"Loaded estimator from {path} with models: {list(self.models.keys())}")
    
    def get_prediction_confidence(self, operator_query: Dict[str, Any]) -> float:
        """Return prediction confidence (std dev across trees)."""
        op_class = operator_query.get('operator_class', 'token')
        
        if op_class not in self.models:
            return 0.0
        
        # Get predictions from all trees
        model = self.models[op_class]
        
        # Extract features (same as predict)
        model_spec = operator_query.get('model_spec', {})
        hw_spec = operator_query.get('hardware', {})
        inputs = operator_query.get('inputs', {})
        phase = operator_query.get('phase', 'both')
        
        if op_class == 'token':
            features = features_for_token_op(model_spec, inputs, hw_spec)
        elif op_class == 'sequence':
            features = features_for_sequence_op(model_spec, inputs, hw_spec, phase)
        elif op_class == 'comm':
            features = features_for_comm_op(inputs, hw_spec)
        else:
            return 0.0
        
        features['tp'] = operator_query.get('tp', 1)
        features['pp'] = operator_query.get('pp', 1)
        features = normalize_features(features)
        
        feature_names = self.feature_info[op_class]
        feature_vec = np.array([features.get(k, 0.0) for k in feature_names])
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(feature_vec.reshape(1, -1))[0] 
                               for tree in model.estimators_])
        
        return float(np.std(predictions))