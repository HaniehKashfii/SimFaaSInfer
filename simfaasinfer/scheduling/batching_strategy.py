"""Batching strategies for different schedulers."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..serverless.function_instance import Request, FunctionInstance
from ..models.model_config import ModelConfig
from ..utils.logger import setup_logger


class BatchingStrategy(ABC):
    """Abstract base class for batching strategies."""

    def __init__(self, config: dict, model_config: ModelConfig, instance: FunctionInstance):
        """Initialize batching strategy.

        Args:
            config: Scheduler configuration
            model_config: Model configuration
            instance: Function instance
        """
        self.config = config
        self.model_config = model_config
        self.instance = instance
        self.logger = setup_logger(self.__class__.__name__)

        self.max_batch_size = config.get('max_batch_size', 256)
        self.max_tokens_per_batch = config.get('max_tokens_per_batch', 4096)

    @abstractmethod
    def form_batch(self, waiting_requests: List[Request],
                   running_requests: List[Request],
                   current_time: float) -> Optional[List[Request]]:
        """Form a batch for execution.

        Args:
            waiting_requests: Requests waiting to be scheduled
            running_requests: Requests currently running
            current_time: Current simulation time

        Returns:
            Batch of requests or None
        """
        pass

    def _count_tokens(self, requests: List[Request], phase: str = 'mixed') -> int:
        """Count total tokens in a batch.

        Args:
            requests: List of requests
            phase: 'prefill', 'decode', or 'mixed'

        Returns:
            Total token count
        """
        total = 0
        for req in requests:
            if phase == 'prefill' or (phase == 'mixed' and not req.prefill_complete):
                total += req.prompt_tokens
            elif phase == 'decode' or (phase == 'mixed' and req.prefill_complete):
                total += 1  # One token per decode request
        return total


class VLLMBatcher(BatchingStrategy):
    """vLLM batching strategy (continuous batching with preemption).

    - Prioritizes prefill over decode
    - Uses continuous batching
    - Can preempt decodes for new prefills
    """

    def form_batch(self, waiting_requests: List[Request],
                   running_requests: List[Request],
                   current_time: float) -> Optional[List[Request]]:
        """Form batch using vLLM strategy.

        Args:
            waiting_requests: Waiting requests
            running_requests: Running requests
            current_time: Current time

        Returns:
            Batch or None
        """
        batch = []
        token_count = 0

        # First, try to add waiting prefills
        prefills = [r for r in waiting_requests if not r.prefill_complete]
        for req in prefills[:self.max_batch_size]:
            if len(batch) >= self.max_batch_size:
                break
            if token_count + req.prompt_tokens > self.max_tokens_per_batch:
                break

            batch.append(req)
            token_count += req.prompt_tokens

        # Then add running decodes if space available
        if len(batch) < self.max_batch_size:
            decodes = [r for r in running_requests if r.prefill_complete and not r.is_complete()]
            for req in decodes:
                if len(batch) >= self.max_batch_size:
                    break
                if token_count + 1 > self.max_tokens_per_batch:
                    break

                batch.append(req)
                token_count += 1

        return batch if batch else None


class OrcaBatcher(BatchingStrategy):
    """Orca batching strategy (iteration-level scheduling).

    - Schedules at iteration granularity
    - No preemption
    - Fair scheduling between prefill and decode
    """

    def form_batch(self, waiting_requests: List[Request],
                   running_requests: List[Request],
                   current_time: float) -> Optional[List[Request]]:
        """Form batch using Orca strategy.

        Args:
            waiting_requests: Waiting requests
            running_requests: Running requests
            current_time: Current time

        Returns:
            Batch or None
        """
        batch = []

        # Add all running requests first (no preemption)
        running_decodes = [
            r for r in running_requests
            if r.prefill_complete and not r.is_complete()
        ]
        batch.extend(running_decodes[:self.max_batch_size])

        # Fill remaining slots with prefills
        if len(batch) < self.max_batch_size:
            prefills = [r for r in waiting_requests if not r.prefill_complete]
            remaining_slots = self.max_batch_size - len(batch)

            token_count = len(batch)  # Decodes use 1 token each
            for req in prefills:
                if len(batch) >= self.max_batch_size:
                    break
                if token_count + req.prompt_tokens > self.max_tokens_per_batch:
                    break

                batch.append(req)
                token_count += req.prompt_tokens

        return batch if batch else None


class SarathiBatcher(BatchingStrategy):
    """Sarathi-Serve batching strategy (chunked prefills).

    - Chunks prefills to avoid pausing decodes
    - Creates hybrid batches with prefill chunks and decodes
    - Minimizes decode latency while maintaining throughput
    """

    def __init__(self, config: dict, model_config: ModelConfig, instance: FunctionInstance):
        """Initialize Sarathi batcher.

        Args:
            config: Configuration
            model_config: Model configuration
            instance: Function instance
        """
        super().__init__(config, model_config, instance)
        self.chunk_size = config.get('chunk_size', 512)

    def form_batch(self, waiting_requests: List[Request],
                   running_requests: List[Request],
                   current_time: float) -> Optional[List[Request]]:
        """Form batch using Sarathi strategy.

        Args:
            waiting_requests: Waiting requests
            running_requests: Running requests
            current_time: Current time

        Returns:
            Batch or None
        """
        batch = []
        token_count = 0

        # Always include running decodes
        running_decodes = [
            r for r in running_requests
            if r.prefill_complete and not r.is_complete()
        ]
        batch.extend(running_decodes)
        token_count += len(running_decodes)

        # Add chunked prefills
        if len(batch) < self.max_batch_size:
            prefills = [r for r in waiting_requests if not r.prefill_complete]

            for req in prefills:
                if len(batch) >= self.max_batch_size:
                    break

                # Add chunk of prefill
                remaining_prefill = req.prompt_tokens - req.kv_cache_tokens
                chunk_tokens = min(remaining_prefill, self.chunk_size)

                if token_count + chunk_tokens > self.max_tokens_per_batch:
                    break

                batch.append(req)
                token_count += chunk_tokens

                # Mark progress (would be handled during execution)
                # This is simplified for simulation

        return batch if batch else None


class FasterTransformerBatcher(BatchingStrategy):
    """FasterTransformer batching strategy (static batching).

    - Fixed batch size
    - Waits to fill batch before executing
    - Separate prefill and decode phases
    """

    def form_batch(self, waiting_requests: List[Request],
                   running_requests: List[Request],
                   current_time: float) -> Optional[List[Request]]:
        """Form batch using FasterTransformer strategy.

        Args:
            waiting_requests: Waiting requests
            running_requests: Running requests
            current_time: Current time

        Returns:
            Batch or None
        """
        # Prefer decode phase
        decodes = [r for r in running_requests if not r.is_complete()]
        if decodes:
            # Fill batch with decodes
            batch = decodes[:self.max_batch_size]
            return batch if len(batch) > 0 else None

        # Otherwise do prefill phase
        prefills = [r for r in waiting_requests if not r.prefill_complete]
        if len(prefills) >= self.max_batch_size // 2:  # Wait for minimum batch
            batch = prefills[:self.max_batch_size]
            return batch

        return None
