# simfaasinfer/optimizer/optimizer_base.py
"""
Abstract base class for optimizers (grid, Bayesian, GA).
"""
from typing import Dict, Any
from abc import ABC, abstractmethod


class OptimizerBase(ABC):
    def __init__(self, search_space: Dict[str, Any]):
        self.search_space = search_space
        self.history = []

    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        """Return a candidate configuration to evaluate."""
        pass

    @abstractmethod
    def observe(self, candidate: Dict[str, Any], metrics: Dict[str, Any]):
        """Update internal state after observing metrics for candidate."""
        pass


class GridSearchOptimizer(OptimizerBase):
    """Simple grid search optimizer."""
    
    def __init__(self, search_space: Dict[str, Any]):
        super().__init__(search_space)
        self.candidates = self._generate_grid()
        self.current_idx = 0
    
    def _generate_grid(self) -> list:
        """Generate grid of configurations."""
        # Similar to vidur_search candidate generation
        candidates = []
        
        for gpu_type in self.search_space.get('gpu_types', ['A100']):
            for num_gpus in self.search_space.get('num_gpus', [1, 2, 4]):
                for tp in self.search_space.get('tp_sizes', [1, 2]):
                    candidates.append({
                        'gpu_type': gpu_type,
                        'num_gpus': num_gpus,
                        'tp': tp,
                        'pp': 1
                    })
        
        return candidates
    
    def suggest(self) -> Dict[str, Any]:
        """Return next candidate from grid."""
        if self.current_idx >= len(self.candidates):
            return None
        
        candidate = self.candidates[self.current_idx]
        self.current_idx += 1
        return candidate
    
    def observe(self, candidate: Dict[str, Any], metrics: Dict[str, Any]):
        """Record observation."""
        self.history.append({
            'candidate': candidate,
            'metrics': metrics
        })