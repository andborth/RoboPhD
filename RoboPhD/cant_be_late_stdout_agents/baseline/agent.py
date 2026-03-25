import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'

    def __init__(self, args):
        super().__init__(args)
        self._first_step = True

    def reset(self, env, task):
        super().reset(env, task)
        self._first_step = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env

        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        remaining_time = self.deadline - env.elapsed_seconds
        slack = remaining_time - remaining_task_time

        # --- Demonstration: print() output is captured as agent_stdout in diagnostics ---
        if self._first_step:
            print(f"Task: duration={self.task_duration/3600:.1f}h, deadline={self.deadline/3600:.1f}h, overhead={self.restart_overhead/3600:.2f}h")
            self._first_step = False

        if remaining_task_time + self.restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
