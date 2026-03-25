from sky_spot.utils import ClusterType


class Agent:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, last_cluster_type, has_spot, elapsed_seconds, gap_seconds,
             restart_overhead, task_duration, deadline, task_done_time):
        remaining_task_time = task_duration - sum(task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        remaining_time = deadline - elapsed_seconds

        if remaining_task_time + restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE
