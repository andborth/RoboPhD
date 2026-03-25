from sky_spot.utils import ClusterType


class Agent:
    def __init__(self):
        self._first_step = True

    def reset(self):
        self._first_step = True

    def step(self, last_cluster_type, has_spot, elapsed_seconds, gap_seconds,
             restart_overhead, task_duration, deadline, task_done_time):
        remaining_task_time = task_duration - sum(task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        remaining_time = deadline - elapsed_seconds
        slack = remaining_time - remaining_task_time

        # --- Demonstration: print() output is captured as agent_stdout in diagnostics ---
        if self._first_step:
            print(f"Task: duration={task_duration/3600:.1f}h, deadline={deadline/3600:.1f}h, overhead={restart_overhead/3600:.2f}h, gap={gap_seconds:.0f}s")
            self._first_step = False

        if remaining_task_time + restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE
