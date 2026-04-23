from ensemble_launcher.config import PolicyConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler import Policy, SchedulerState, policy_registry


@policy_registry.register(policy_name="shortest_first")
class ShortestFirst(Policy):
    def __init__(self, policy_config=PolicyConfig(), logger=None):
        super().__init__(policy_config, logger)

    def get_score(self, task: Task, scheduler_state: SchedulerState = None):
        self.logger.info("Using shortest first policy")
        duration = task.estimated_runtime
        return -(duration**2)


@policy_registry.register(policy_name="longest_first")
class LongestFirst(Policy):
    def __init__(self, policy_config=PolicyConfig(), logger=None):
        super().__init__(policy_config, logger)

    def get_score(self, task: Task, scheduler_state: SchedulerState = None):
        self.logger.info("Using longest first policy")
        duration = task.estimated_runtime
        return duration**2


@policy_registry.register(policy_name="largest_first")
class LargestFirst(Policy):
    def __init__(self, policy_config=PolicyConfig(), logger=None):
        super().__init__(policy_config, logger)

    def get_score(self, task: Task, scheduler_state: SchedulerState = None):
        self.logger.info("Using largest first policy")
        duration = task.estimated_runtime
        np = task.nnodes * task.ppn
        return (duration * np) ** 2
