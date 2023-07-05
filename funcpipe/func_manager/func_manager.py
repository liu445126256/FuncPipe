'''function manager is in charge of launch and maintain the functions'''
from typing import Dict
import threading

from funcpipe.platforms import Platform
from funcpipe.planner import PipelineArch
from funcpipe.debugger import Logger

class FunctionManager:
    def __init__(self):
        pass

    def start_pipeline(self, pipeline_arch: PipelineArch, launch_info):
        rank_num = pipeline_arch.get_rank_n()
        if Platform.platform_type == 'local':
            for rank in range(1, rank_num):
                Logger.debug("Local: Launching rank %d" % rank)
                launch_info["rank"] = rank
                Platform.invoke(launch_info, asynchronous = True)
        else:
            function_name = launch_info["function_name"]
            for rank in range(rank_num):
                mem_allocation = pipeline_arch.get_rank_mem(rank)
                Logger.debug("Launching rank {}, mem size: {} MB".format(rank, mem_allocation))
                launch_info["rank"] = rank
                launch_info["is_init_worker"] = 0
                launch_info["function_name"] =  function_name + "_{}m".format(mem_allocation)
                Platform.invoke(launch_info, asynchronous = True)
