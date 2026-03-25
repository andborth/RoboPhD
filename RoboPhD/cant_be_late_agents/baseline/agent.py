from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'

    def __init__(self, args):
        super().__init__(args)

    def reset(self, env, task):
        super().reset(env, task)
        self.elapsed_seconds = env.elapsed_seconds
        self.gap_seconds = env.gap_seconds

    def step(self):
        self.elapsed_seconds = self.env.elapsed_seconds
        return super().step()

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        remaining_time = self.deadline - self.elapsed_seconds
        slack = remaining_time - remaining_task_time

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
