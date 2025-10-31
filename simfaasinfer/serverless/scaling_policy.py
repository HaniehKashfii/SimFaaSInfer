"""Auto-scaling policies for serverless instances."""

from typing import Dict, Optional
from enum import Enum

from ..utils.logger import setup_logger


class ScalingDecision(Enum):
    """Scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


class ScalingPolicy:
    """Auto-scaling policy for serverless LLM inference.

    Makes decisions based on utilization metrics and configured thresholds.
    """

    def __init__(self, config: Dict):
        """Initialize scaling policy.

        Args:
            config: Scaling configuration
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        self.enabled = config.get('enabled', True)
        self.min_instances = config.get('min_instances', 1)
        self.max_instances = config.get('max_instances', 10)

        self.scale_up_threshold = config.get('scale_up_threshold', 0.8)
        self.scale_down_threshold = config.get('scale_down_threshold', 0.2)

        self.cooldown_period = config.get('cooldown_period', 300)

        # State
        self.last_scale_time = 0.0
        self.consecutive_high_util = 0
        self.consecutive_low_util = 0

    def decide(self, current_instances: int, metrics: Dict,
               current_time: float) -> Optional[str]:
        """Make scaling decision based on metrics.

        Args:
            current_instances: Current number of instances
            metrics: Current metrics dictionary
            current_time: Current simulation time

        Returns:
            Scaling decision: 'scale_up', 'scale_down', or None
        """
        if not self.enabled:
            return None

        # Check cooldown period
        if (current_time - self.last_scale_time) < self.cooldown_period:
            return None

        # Get utilization
        utilization = metrics.get('mfu', 0.0)

        # Scale up decision
        if utilization > self.scale_up_threshold:
            self.consecutive_high_util += 1
            self.consecutive_low_util = 0

            # Scale up if consistently high
            if self.consecutive_high_util >= 2 and current_instances < self.max_instances:
                self.last_scale_time = current_time
                self.consecutive_high_util = 0
                self.logger.info(
                    f"Scaling up: utilization={utilization:.2%} > {self.scale_up_threshold:.2%}"
                )
                return 'scale_up'

        # Scale down decision
        elif utilization < self.scale_down_threshold:
            self.consecutive_low_util += 1
            self.consecutive_high_util = 0

            # Scale down if consistently low
            if self.consecutive_low_util >= 3 and current_instances > self.min_instances:
                self.last_scale_time = current_time
                self.consecutive_low_util = 0
                self.logger.info(
                    f"Scaling down: utilization={utilization:.2%} < {self.scale_down_threshold:.2%}"
                )
                return 'scale_down'

        else:
            # Reset counters
            self.consecutive_high_util = 0
            self.consecutive_low_util = 0

        return None