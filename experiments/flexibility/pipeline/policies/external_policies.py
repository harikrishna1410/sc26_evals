from typing import Dict, List, Optional

import numpy as np
from ensemble_launcher.comm.messages import Status
from ensemble_launcher.config import PolicyConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler import ChildrenPolicy, policy_registry
from ensemble_launcher.scheduler.policy import FixedLeafNodePolicy
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList
from ensemble_launcher.scheduler.state import ChildrenAssignment


@policy_registry.register(policy_name="resource_split_policy", type="children_policy")
class ResourceSplitPolicy(ChildrenPolicy):
    def __init__(self, policy_config=PolicyConfig(), node_id=None, logger=None):
        super().__init__(policy_config, node_id, logger)
        self.logger.info(f"Using ResourceSplitPolicy")

    def get_children_resources(
        self, tasks: Dict[str, Task], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        sim_nodes = self.policy_config.sim_nodes
        nodes_per_sim = self.policy_config.nodes_per_sim
        node_names = nodes.nodes
        resources = nodes.resources
        self.logger.info(f"sim_nodes = {sim_nodes}, nodes_per_sim = {nodes_per_sim}")

        children = {}
        if level == 0:
            # Simulation + retrain nodes
            children[0] = JobResource(
                nodes=node_names[:sim_nodes],
                resources=[
                    NodeResourceList(cpus=node.cpus[:13], gpus=node.gpus)
                    for node in resources[:sim_nodes]
                ],
            )

            # Pre/Post processing nodes = CPU only
            children[1] = JobResource(
                nodes=node_names,
                resources=[NodeResourceList(cpus=node.cpus[13:]) for node in resources],
            )

            # inference nodes
            children[2] = JobResource(
                nodes=node_names[sim_nodes:],
                resources=[
                    NodeResourceList(cpus=node.cpus[:13], gpus=node.gpus)
                    for node in resources[sim_nodes:]
                ],
            )
        else:
            if self.node_id == "main.m2":
                nchildren = len(nodes.nodes)
            elif self.node_id == "main.m1":
                nchildren = len(nodes.nodes)
            else:
                nchildren = int(len(nodes.nodes) // nodes_per_sim)

            nodes_per_child = int(len(nodes.nodes) // nchildren)
            rem = len(nodes.nodes) % nodes_per_child
            child_id = 0
            for child_id in range(nchildren):
                children[child_id] = JobResource(
                    nodes=node_names[
                        child_id * nodes_per_child : (child_id + 1) * nodes_per_child
                    ],
                    resources=resources[
                        child_id * nodes_per_child : (child_id + 1) * nodes_per_child
                    ],
                )
            if rem > 0:
                if len(children) == 0:
                    children[child_id] = JobResource(
                        nodes=node_names, resources=resources
                    )
                else:
                    children[child_id].nodes.extend(node_names[-rem:])
                    children[child_id].resources.extend(resources[-rem:])
        return children

    def get_children_tasks(
        self,
        tasks: Dict[str, Task],
        children_resources: Dict[int, JobResource],
        ntask: Optional[int] = None,
        child_assignments: Optional[Dict[str, "ChildrenAssignment"]] = None,
        child_status: Optional[Dict[int, "Status"]] = None,
        level: Optional[int] = None,
        **kwargs,
    ):
        assert level is not None, "Need level to distribute tasks"

        worker_ids = list(children_resources.keys())
        nchildren = len(worker_ids)
        wid_to_task_id_map: Dict[int, List[str]] = {wid: [] for wid in worker_ids}
        task_id_to_wid_map: Dict[str, int] = {}
        removed_tasks: List[str] = []
        if level == 0:
            for task_id, task in tasks.items():
                if task.tag == "sim" or task.tag == "training":
                    wid_to_task_id_map[0].append(task_id)
                    task_id_to_wid_map[task_id] = 0
                elif task.tag == "post" or task.tag == "pipeline":
                    wid_to_task_id_map[1].append(task_id)
                    task_id_to_wid_map[task_id] = 1
                else:
                    wid_to_task_id_map[2].append(task_id)
                    task_id_to_wid_map[task_id] = 2
        else:
            ##Assign in round robin fashion
            worker_task_counts = {wid: 0 for wid in worker_ids}
            current = worker_ids.index(
                min(
                    worker_ids,
                    key=lambda worker_id: len(child_assignments[worker_id]["task_ids"]),
                )
            )

            for task_id, task in tasks.items():
                task_resource = task.get_resource_requirements()
                assigned = False

                for attempt in range(nchildren):
                    wid = worker_ids[(current + attempt) % nchildren]

                    if ntask is not None and worker_task_counts[wid] >= ntask:
                        continue

                    if task_resource in children_resources[wid]:
                        wid_to_task_id_map[wid].append(task_id)
                        task_id_to_wid_map[task_id] = wid
                        worker_task_counts[wid] += 1
                        current = (worker_ids.index(wid) + 1) % nchildren
                        assigned = True
                        break
                if not assigned:
                    removed_tasks.append(task_id)
        return wid_to_task_id_map, task_id_to_wid_map, removed_tasks


@policy_registry.register(policy_name="smart_split_policy", type="children_policy")
class SmartResourceSplitPolicy(ChildrenPolicy):
    def __init__(self, policy_config=PolicyConfig(), node_id=None, logger=None):
        super().__init__(policy_config, node_id, logger)
        self.logger.info(f"Using ResourceSplitPolicy")

    def get_children_resources(
        self, tasks: Dict[str, Task], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        sim_nodes = self.policy_config.sim_nodes
        nodes_per_sim = self.policy_config.nodes_per_sim
        node_names = nodes.nodes
        resources = nodes.resources
        self.logger.info(f"sim_nodes = {sim_nodes}, nodes_per_sim = {nodes_per_sim}")

        children = {}
        if level == 0:
            # Simulation + retrain nodes
            children[0] = JobResource(
                nodes=node_names[:8],
                resources=[
                    NodeResourceList(cpus=node.cpus[:13], gpus=node.gpus)
                    for node in resources[:8]
                ],
            )

            # Pre/Post processing nodes = CPU only
            children[1] = JobResource(
                nodes=node_names,
                resources=[NodeResourceList(cpus=node.cpus[13:]) for node in resources],
            )

            # inference nodes
            children[2] = JobResource(
                nodes=node_names[8:16],
                resources=[
                    NodeResourceList(cpus=node.cpus[:13], gpus=node.gpus)
                    for node in resources[8:16]
                ],
            )

            # Sim + inference nodes
            children[3] = JobResource(
                nodes=node_names[16:],
                resources=[
                    NodeResourceList(cpus=node.cpus[:13], gpus=node.gpus)
                    for node in resources[16:]
                ],
            )
        else:
            if self.node_id == "main.m3":
                nchildren = int(len(nodes.nodes) // nodes_per_sim)
            elif self.node_id == "main.m2":
                nchildren = len(nodes.nodes)
            elif self.node_id == "main.m1":
                nchildren = len(nodes.nodes)
            else:
                nchildren = int(len(nodes.nodes) // nodes_per_sim)

            nodes_per_child = int(len(nodes.nodes) // nchildren)
            rem = len(nodes.nodes) % nodes_per_child
            child_id = 0
            for child_id in range(nchildren):
                children[child_id] = JobResource(
                    nodes=node_names[
                        child_id * nodes_per_child : (child_id + 1) * nodes_per_child
                    ],
                    resources=resources[
                        child_id * nodes_per_child : (child_id + 1) * nodes_per_child
                    ],
                )
            if rem > 0:
                if len(children) == 0:
                    children[child_id] = JobResource(
                        nodes=node_names, resources=resources
                    )
                else:
                    children[child_id].nodes.extend(node_names[-rem:])
                    children[child_id].resources.extend(resources[-rem:])
        return children

    def get_children_tasks(
        self,
        tasks: Dict[str, Task],
        children_resources: Dict[int, JobResource],
        ntask: Optional[int] = None,
        child_assignments: Optional[Dict[str, "ChildrenAssignment"]] = None,
        child_status: Optional[Dict[int, "Status"]] = None,
        level: Optional[int] = None,
        **kwargs,
    ):
        assert level is not None, "Need level to distribute tasks"

        worker_ids = list(children_resources.keys())
        nchildren = len(worker_ids)
        wid_to_task_id_map: Dict[int, List[str]] = {wid: [] for wid in worker_ids}
        task_id_to_wid_map: Dict[str, int] = {}
        removed_tasks: List[str] = []
        nsims = 0
        ninf = 0
        if level == 0:
            for task_id, task in tasks.items():
                if task.tag == "sim":
                    if nsims % 2 == 0:
                        wid_to_task_id_map[0].append(task_id)
                        task_id_to_wid_map[task_id] = 0
                    else:
                        wid_to_task_id_map[3].append(task_id)
                        task_id_to_wid_map[task_id] = 3
                    nsims += 1
                elif task.tag == "post":
                    wid_to_task_id_map[1].append(task_id)
                    task_id_to_wid_map[task_id] = 1
                else:
                    if ninf % 2 == 0:
                        wid_to_task_id_map[2].append(task_id)
                        task_id_to_wid_map[task_id] = 2
                    else:
                        wid_to_task_id_map[3].append(task_id)
                        task_id_to_wid_map[task_id] = 3
                    ninf += 1
        else:
            ##Assign in round robin fashion
            worker_task_counts = {wid: 0 for wid in worker_ids}
            current = worker_ids.index(
                min(
                    worker_ids,
                    key=lambda worker_id: len(child_assignments[worker_id]["task_ids"]),
                )
            )

            for task_id, task in tasks.items():
                task_resource = task.get_resource_requirements()
                assigned = False

                for attempt in range(nchildren):
                    wid = worker_ids[(current + attempt) % nchildren]

                    if ntask is not None and worker_task_counts[wid] >= ntask:
                        continue

                    if task_resource in children_resources[wid]:
                        wid_to_task_id_map[wid].append(task_id)
                        task_id_to_wid_map[task_id] = wid
                        worker_task_counts[wid] += 1
                        current = (worker_ids.index(wid) + 1) % nchildren
                        assigned = True
                        break
                if not assigned:
                    removed_tasks.append(task_id)
        return wid_to_task_id_map, task_id_to_wid_map, removed_tasks


@policy_registry.register(policy_name="routing_policy", type="children_policy")
class SmartRoutingPolicy(FixedLeafNodePolicy):
    def __init__(self, policy_config=PolicyConfig(), node_id=None, logger=None):
        super().__init__(policy_config, node_id, logger)
        self.logger.info(f"Using smart routing policy")
        self._assigned_cpus = None
        self._assigned_gpus = None

    def get_children_resources(
        self, tasks: Dict[str, Task], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        ret = super().get_children_resources(tasks, nodes, level)
        self._assigned_cpus = np.zeros(len(ret))
        self._assigned_gpus = np.zeros(len(ret))
        return ret

    def get_children_tasks(
        self,
        tasks: Dict[str, Task],
        children_resources: Dict[int, JobResource],
        ntask: Optional[int] = None,
        child_assignments: Optional[Dict[int, "ChildrenAssignment"]] = None,
        child_status: Optional[Dict[int, "Status"]] = None,
        level: Optional[int] = None,
        **kwargs,
    ):
        worker_ids = list(children_resources.keys())
        wid_to_task_id_map: Dict[int, List[str]] = {wid: [] for wid in worker_ids}
        task_id_to_wid_map: Dict[str, int] = {}
        removed_tasks: List[str] = []

        for task_id, task in tasks.items():
            ncpus = task.nnodes * task.ppn
            ngpus = ncpus * task.ngpus_per_process

            if ngpus > 0:
                wid = np.argmin(self._assigned_gpus)
            else:
                wid = np.argmin(self._assigned_cpus)

            wid_to_task_id_map[wid].append(task_id)
            task_id_to_wid_map[task_id] = wid
            self._assigned_gpus[wid] += ngpus
            self._assigned_cpus[wid] += ncpus

        return wid_to_task_id_map, task_id_to_wid_map, removed_tasks
