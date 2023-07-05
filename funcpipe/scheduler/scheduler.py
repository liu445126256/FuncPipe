'''Scheduler generates all tasks for one pipeline run
The dependencies between tasks are specified
Todo: For now schedule() is called for every batch, it should only be called once if the pipeline stays unchanged
'''
from typing import Deque, Dict, Tuple, List
from collections import deque

from funcpipe.scheduler.task import Task
from funcpipe.planner import PipelineArch
from funcpipe.debugger import Logger

class ScheduleInfo:
    def __init__(self, pipeline_arch: PipelineArch, my_rank: int, my_stage_id: int,
                 my_tensor_id: int, my_data_id: int, microbatch_num: int, skip_links: List):
        self.pipline_arch = pipeline_arch
        self.my_rank = my_rank
        self.my_stage_id = my_stage_id
        self.my_tensor_id = my_tensor_id
        self.my_data_id = my_data_id
        self.micro_num = microbatch_num
        self.skip_links = skip_links

class Scheduler:
    def __init__(self):
        self.global_id = 0
        self.round_robin_pointer = []
        self.round_robin_bound = []

    def schedule(self, schedule_info: ScheduleInfo) -> Deque[Task]:
        scheduled_tasks: Deque[Task] = deque()
        # '''schedule the pipline here'''

        # GPIPE (we add data parallelism to the pipeline, we assume data parallelism is the same for each stage)
        # todo: Note that the implementation here supports tensor parallelism but tensor parallelism is not supported by other modules for now
        # Total num of tasks: micro_batch_num * (3 + 3) - 2 - 2 -> forward, backward and their communication ops
        total_stage_n = schedule_info.pipline_arch.get_stage_n()
        self.round_robin_pointer = [0 for i in range(total_stage_n)]
        self.round_robin_bound = [0 for i in range(total_stage_n)]
        all_batch_paths = []
        for batch_id in range(schedule_info.micro_num):
            # compute the path for this micro batch
            batch_path = []
            for stage_id in range(total_stage_n):
                data_parallel_n = schedule_info.pipline_arch.get_stage_data_n(stage_id)
                tensor_parallel_n = schedule_info.pipline_arch.get_stage_tensor_n(stage_id)
                self.round_robin_bound[stage_id] = data_parallel_n

                # determine ranks to process this batch
                rank_ids = []
                data_id = self.round_robin_pointer[stage_id]
                self.round_robin_pointer[stage_id] = (self.round_robin_pointer[stage_id] + 1) % self.round_robin_bound[stage_id]
                for tensor_id in range(tensor_parallel_n):
                    rank_ids.append(schedule_info.pipline_arch.arch_id_to_rank_id(stage_id, tensor_id, data_id))
                batch_path.append(rank_ids)

            # add task
            # fwd
            all_batch_paths.append(batch_path)
            for stage_id, rank_ids in enumerate(batch_path):
                if schedule_info.my_rank in rank_ids:
                    assert schedule_info.my_stage_id == stage_id
                    recv_act_id = -1
                    if schedule_info.my_stage_id != 0:
                        # '''recv activation'''
                        task = self.get_task(Task.RECV_INPUT)
                        task.set_micro_id(batch_id)
                        comm_rank_ids: List[List, int] = []
                        comm_rank_ids.extend([batch_path[stage_id - 1], schedule_info.my_rank])
                        task.config_comm_task(comm_rank_ids)
                        scheduled_tasks.append(task)
                        recv_act_id = task.id

                    # '''fwd computation'''
                    task = self.get_task(Task.FORWARD)
                    task.set_micro_id(batch_id)
                    task.config_comp_task(schedule_info.my_rank, batch_id)
                    scheduled_tasks.append(task)
                    if recv_act_id >= 0: task.add_dependency(recv_act_id)
                    fwd_id = task.id

                    if schedule_info.my_stage_id != (schedule_info.pipline_arch.get_stage_n() - 1):
                        # '''send activation'''
                        task = self.get_task(Task.SEND_OUTPUT)
                        task.set_micro_id(batch_id)
                        comm_rank_ids: List[int, List] = []
                        comm_rank_ids.extend([schedule_info.my_rank, batch_path[stage_id + 1]])
                        task.config_comm_task(comm_rank_ids)
                        scheduled_tasks.append(task)
                        task.add_dependency(fwd_id)

        Logger.debug("Batch paths: %s" % str(all_batch_paths))

        for batch_id in range(len(all_batch_paths) - 1, -1, -1):
            batch_path = all_batch_paths[batch_id]
            # bp
            for stage_id, rank_ids in enumerate(batch_path):
                if schedule_info.my_rank in rank_ids:
                    assert schedule_info.my_stage_id == stage_id
                    recv_grad_id = -1
                    if schedule_info.my_stage_id != (schedule_info.pipline_arch.get_stage_n() - 1):
                        # '''recv gradient'''
                        task = self.get_task(Task.RECV_GRAD)
                        task.set_micro_id(batch_id)
                        comm_rank_ids: List[List, int] = []
                        comm_rank_ids.extend([batch_path[stage_id + 1], schedule_info.my_rank])
                        task.config_comm_task(comm_rank_ids)
                        scheduled_tasks.append(task)
                        recv_grad_id = task.id

                    # '''backward computation'''
                    task = self.get_task(Task.BACKWARD)
                    task.set_micro_id(batch_id)
                    task.config_comp_task(schedule_info.my_rank, batch_id)
                    scheduled_tasks.append(task)
                    if recv_grad_id >= 0: task.add_dependency(recv_grad_id)
                    bwd_id = task.id

                    if schedule_info.my_stage_id != 0:
                        # '''send gradient'''
                        task = self.get_task(Task.SEND_GRAD)
                        task.set_micro_id(batch_id)
                        comm_rank_ids: List[int, List] = []
                        comm_rank_ids.extend([schedule_info.my_rank, batch_path[stage_id - 1]])
                        task.config_comm_task(comm_rank_ids)
                        scheduled_tasks.append(task)
                        task.add_dependency(bwd_id)

        # allreduce
        comm_rank_ids: List[int, ...] = schedule_info.pipline_arch.get_data_parallel_group(schedule_info.my_rank)
        allreduce_grad_id = -1
        if len(comm_rank_ids) > 1:
            task = self.get_task(Task.SYNC_WEIGHTS)
            task.config_comm_task(comm_rank_ids)
            scheduled_tasks.append(task)
            allreduce_grad_id = task.id

        # weight update
        task = self.get_task(Task.WEIGHT_UPDATE)
        task.config_comp_task(schedule_info.my_rank, batch_id = -1)
        scheduled_tasks.append(task)
        if allreduce_grad_id >= 0: task.add_dependency(allreduce_grad_id)


        # Change tasks sequence provisionally.
        # Here just change all recv tasks ahead.
        # TODO: change the scheduler's logic fundamentally
        if scheduled_tasks[-2].type == Task.SYNC_WEIGHTS:
            half_proc_length = int((len(scheduled_tasks) - 2 )/ 2)
        else:
            half_proc_length = int((len(scheduled_tasks) - 1) / 2)
        middle_fw = 1 if scheduled_tasks[0].type == Task.RECV_INPUT else 0
        middle_bw = 1 if scheduled_tasks[half_proc_length].type == Task.RECV_GRAD else 0
        new_task_list = deque()
        if middle_fw:
            # add recv tasks first
            for i, j in enumerate(scheduled_tasks):
                if i == half_proc_length:
                    break
                if j.type == Task.RECV_INPUT:
                    new_task_list.append(j)
            for i, j in enumerate(scheduled_tasks):
                if i == half_proc_length:
                    break
                if j.type != Task.RECV_INPUT:
                    new_task_list.append(j)
        else:
            for i, j in enumerate(scheduled_tasks):
                if i == half_proc_length:
                    break
                new_task_list.append(j)
        if middle_bw:
            # add recv tasks first
            for i, j in enumerate(scheduled_tasks):
                if i < half_proc_length:
                    continue
                if j.type == Task.RECV_GRAD:
                    new_task_list.append(j)
            for i, j in enumerate(scheduled_tasks):
                if i < half_proc_length:
                    continue
                if j.type != Task.RECV_GRAD:
                    new_task_list.append(j)
        else:
            for i, j in enumerate(scheduled_tasks):
                if i < half_proc_length:
                    continue
                new_task_list.append(j)

        # return scheduled_tasks
        return new_task_list


    def get_task(self, task_type):
        task = Task(self.global_id, task_type)
        self.global_id += 1
        return task