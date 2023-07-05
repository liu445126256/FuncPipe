'''Planner for pipeline'''
from typing import List, Tuple, Dict
from collections import OrderedDict

from torch import nn

from funcpipe.planner.profiler import Profiler
from funcpipe.planner.pipeline_arch import PipelineArch
from funcpipe.debugger import Logger
from funcpipe.platforms import Platform

class Planner:
    def __init__(self):
        # we manually specify the partition for test
        '''
        self.partition_plan = [1, 2, 3, 1]
        self.tensor_parallelism = [1, 1, 1, 1]
        self.data_parallelism = [1, 1, 1, 1]
        '''
        self.partition_plan = [304]
        self.tensor_parallelism = [1]
        self.data_parallelism = [1]
        self.mem_allocation = [1024]
        self.micro_batchsize = 1

    def plan(self, model:nn.Sequential, model_info: Dict, my_rank, skip_links) -> Tuple[nn.Sequential, PipelineArch, int]:
        '''
        Pipeline design logic
        :param model: user input model, rank id, skip link specification
        :return: model partition for the current
                a dict that specifies the pipeline architecture { stage_id: (start_layer_index, end_layer_index, tensor_parallelism_level, data_parallelism_level)}
        '''
        # partition policy
        # for now we try to balance the memory consumption of each stage (50 MB for each stage)
        # we roughly estimate the memory consumption of the model as batch_size * partition_size * 0.8 -> decides the level of data parallelism
        '''
        MAX_SIZE_THRESH = 10
        MIN_SIZE_THRESH = 2
        batchsize = 64
        self.micro_batchsize = 4
        model_info = Profiler.profile(model)
        layer_partition = []
        partition_size = []
        accum_size= 0
        accum_num = 0
        #print(model_info)
        for i in range(len(model_info.keys())):
            layer_size = model_info[i][0]
            accum_size += layer_size
            accum_num += 1
            if accum_size > MAX_SIZE_THRESH:
                layer_partition.append(accum_num)
                partition_size.append(accum_size)
                accum_size = 0
                accum_num = 0
        if accum_size != 0:
            if accum_size > MIN_SIZE_THRESH:
                partition_size.append(accum_size)
                layer_partition.append(accum_num)
            else:
                partition_size[-1] += accum_size
                layer_partition[-1] += accum_num
        self.tensor_parallelism = [1 for i in layer_partition]
        data_parallel = []
        print(partition_size)
        for i, size in enumerate(partition_size):
            data_level = 1
            while batchsize / data_level * size > 1000:
                data_level *= 2
            data_parallel.append(data_level)
        self.data_parallelism = data_parallel
        self.partition_plan = layer_partition
        Logger.debug("---------------------------------------------")
        Logger.debug("Partition plan:")
        Logger.debug("layer partition: {}".format(str(self.partition_plan)))
        Logger.debug("data parallelism: {}".format(str(self.data_parallelism)))
        Logger.debug("tensor parallelism: {}".format(str(self.tensor_parallelism)))
        Logger.debug("---------------------------------------------")
        exit()
'''
        # generate pipeline arch
        pipeline_arch = PipelineArch()
        start_layer_id = 0
        for i, layer_num in enumerate(self.partition_plan):
            t_para_num = self.tensor_parallelism[i]
            d_para_num = self.data_parallelism[i]
            if Platform.platform_type == "local": mem_size = 0
            else: mem_size = self.mem_allocation[i]
            pipeline_arch.add_stage(start_layer_id, start_layer_id + layer_num - 1, t_para_num, d_para_num, mem_size)
            start_layer_id += layer_num

        # todo: skip connection
        '''
        my_partition_id = pipeline_arch.rank_id_to_partition_id(my_rank)
        Logger.debug("my partition id: %d" % my_partition_id)
        layers = OrderedDict()
        my_partition: List[nn.Sequential] = []

        current_partition_id = 0
        current_partition_size = 0
        layer_id = 0
        for name, layer in model.named_children():
            if current_partition_id > my_partition_id: break
            if current_partition_id == my_partition_id:
                layers[name] = layer
                # if skip connection happens here then split the layer
                for skip in skip_links:
                    if layer_id == skip[0]:
                        my_partition.append(nn.Sequential(layers))
                        layers = OrderedDict()
                        break
            current_partition_size += 1
            if current_partition_size == self.partition_plan[current_partition_id]:
                current_partition_id += 1
                current_partition_size = 0
            layer_id += 1
        if len(layers) > 0: my_partition.append(nn.Sequential(layers))
        '''

        my_partition_id = pipeline_arch.rank_id_to_partition_id(my_rank)
        Logger.debug("my partition id: %d" % my_partition_id)
        layers = OrderedDict()

        current_partition_id = 0
        current_partition_size = 0
        for name, layer in model.named_children():
            if current_partition_id > my_partition_id: break
            if current_partition_id == my_partition_id: layers[name] = layer
            current_partition_size += 1

            if current_partition_size == self.partition_plan[current_partition_id]:
                current_partition_id += 1
                current_partition_size = 0
        my_partition = nn.Sequential(layers)

        return my_partition, pipeline_arch, self.micro_batchsize

    def total_node_num(self, pipeline_arch: Dict) -> int:
        node_num = 0
        for stage_id in pipeline_arch.keys():
            for tensor_id in range(pipeline_arch[stage_id][2]):
                node_num += pipeline_arch[stage_id][3]
        return node_num

    #todo: move to PipelineArch class
    #'''find partition id basing on rank id'''
    def rank_id_to_partition_id(self, rank_id, pipeline_arch: PipelineArch) -> int:
        partition_id = 0
        node_num = 0
        for stage_id in range(pipeline_arch.get_stage_n()):
            for tensor_id in range(pipeline_arch.get_stage_tensor_n(stage_id)):
                if (node_num + pipeline_arch.get_stage_data_n(stage_id)) >= (rank_id + 1):
                    return partition_id
                node_num += pipeline_arch.get_stage_data_n(stage_id)
                partition_id += 1
        raise Exception("Failed to map rank to partition!")

    #todo: move to PipelineArch class
    def rank_id_to_arch_ids(self, rank_id, pipeline_arch: PipelineArch) -> Tuple[int, int, int]: #[stage_id, tensor_id, data_id]
        s_id = -1
        t_id = -1
        d_id = -1
        node_num = 0
        for stage_id in range(pipeline_arch.get_stage_n()):
            for tensor_id in range(pipeline_arch.get_stage_tensor_n(stage_id)):
                if (node_num + pipeline_arch.get_stage_data_n(stage_id)) >= (rank_id + 1):
                    s_id = stage_id
                    t_id = tensor_id
                    d_id = rank_id - node_num
                node_num += pipeline_arch.get_stage_data_n(stage_id)

        assert s_id >= 0 and t_id >= 0 and d_id >= 0

        return s_id, t_id, d_id
