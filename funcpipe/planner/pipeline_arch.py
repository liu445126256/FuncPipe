from typing import Tuple

class PipelineArch:
    def __init__(self):
        self.pipeline_arch = {}
        self.stage_id = 0
        self.rank_num = 0

        # current rank info
        # todo: skip links

    def __str__(self):
        return str(self.pipeline_arch)

    def add_stage(self, layer_index_start, layer_index_end, tensor_parallel_n, data_parallel_n, memory_allocation):
        self.pipeline_arch[self.stage_id] = (layer_index_start, layer_index_end, tensor_parallel_n, data_parallel_n, memory_allocation)
        self.stage_id += 1
        self.rank_num += (tensor_parallel_n * data_parallel_n)

    def get_stage_data_n(self, stage_id):
        return self.pipeline_arch[stage_id][3]

    def get_stage_tensor_n(self, stage_id):
        return self.pipeline_arch[stage_id][2]

    def get_stage_n(self):
        return self.stage_id

    def get_rank_n(self):
        return self.rank_num

    def get_rank_mem(self, rank_id):
        stage_id, _, _ = self.rank_id_to_arch_ids(rank_id)
        mem_allocation = self.pipeline_arch[stage_id][4]
        return mem_allocation

    def arch_id_to_rank_id(self, stage_id, tensor_id, data_id):
        rank_id = 0
        for i in range(stage_id):
            rank_id += self.get_stage_tensor_n(i) * self.get_stage_data_n(i)

        rank_id += (tensor_id * self.get_stage_data_n(stage_id))
        rank_id += data_id

        return rank_id

    def rank_id_to_arch_ids(self, rank_id) -> Tuple[int, int, int]:  # [stage_id, tensor_id, data_id]
        node_num = 0
        for stage_id in range(self.get_stage_n()):
            for tensor_id in range(self.get_stage_tensor_n(stage_id)):
                if (node_num + self.get_stage_data_n(stage_id)) >= (rank_id + 1):
                    s_id = stage_id
                    t_id = tensor_id
                    d_id = rank_id - node_num
                    assert s_id >= 0 and t_id >= 0 and d_id >= 0
                    return s_id, t_id, d_id

                node_num += self.get_stage_data_n(stage_id)
        raise Exception("rank to arch id failed!")

    # '''find partition id basing on rank id (multiple ranks can have the same partition ID due to data parallelism)'''
    def rank_id_to_partition_id(self, rank_id) -> int:
        partition_id = 0
        node_num = 0
        for stage_id in range(self.get_stage_n()):
            for tensor_id in range(self.get_stage_tensor_n(stage_id)):
                if (node_num + self.get_stage_data_n(stage_id)) >= (rank_id + 1):
                    return partition_id
                node_num += self.get_stage_data_n(stage_id)
                partition_id += 1
        raise Exception("Failed to map rank to partition!")

    def get_data_parallel_group(self, rank_id):
        group_ids = []
        stage_id, tensor_id, data_id = self.rank_id_to_arch_ids(rank_id)
        for member_data_id in range(self.get_stage_data_n(stage_id)):
            group_ids.append(self.arch_id_to_rank_id(stage_id, tensor_id, member_data_id))
        return group_ids