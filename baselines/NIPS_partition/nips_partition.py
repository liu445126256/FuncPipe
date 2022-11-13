from baselines.NIPS_partition._data_translate import translate_model_info, ali_env, dnn_model
from baselines.NIPS_partition.ip import ip_solve

from pipetests.nsdi_paper.testbed_model import ali_env, aws_env
from pipetests.nsdi_paper.performance_model import pipeline_cost
from pipetests.nsdi_paper.performance_model import policy as plc


def nips_partition(model_info, env, memory_size, max_node_num, batch_size, micro_batch_num):
    graph = translate_model_info(model_info, env, memory_size, max_node_num, batch_size, micro_batch_num)
    result = ip_solve(graph)

    return result

# translate results format
def translate_results(model, nips_result, mem_allocation):
    compressed_layer_info = model.get_compression_info()
    total_layer_num = len(compressed_layer_info)
    partition = [0 for i in range(total_layer_num)]
    '''
    start_id = 0
    while start_id < total_layer_num:
        for node in nips_result["fpgas"]:
            indexes = node["nodes"]
            if start_id in indexes:
                start_id += 1
    '''
    for node in nips_result["fpgas"]:
        indexes = node["nodes"]
        if len(indexes) > 0:
            assert len(indexes) % 2 == 0
            partition[indexes[-2]] = 1

    partition_plan = []
    data_parallelism = []
    resource_type = []

    layer_sum = 0
    for li, split in enumerate(partition):
        layer_sum += 1
        if partition[li] == 1:
            partition_plan.append(layer_sum)
            data_parallelism.append(1)
            resource_type.append(mem_allocation)
            layer_sum = 0
    p = plc(partition=partition_plan, data_parallel=data_parallelism, resource_type=resource_type, env=env)

    print("partition_plan = " + str(partition_plan))
    print("data_parallelism =" + str(data_parallelism))
    print("resource_type = " + str(resource_type))

    return p

if __name__ == "__main__":
    '''
    import pickle
    f = open("../../tools/resnet_101", "rb")
    #f = open("../../tools/bert-base", "rb")
    model_info, base_mem = pickle.load(f)
    model = dnn_model(model_info, base_mem, max_layer_num = 30)
    '''
    import time
    start_t = time.time()

    opt_result = {}

    env = aws_env
    #env.MODEL_SYC_MEM_FACTOR = 1
    model = env.load_resnet101(max_layer_num=30, merge_method="balance-computation")
    max_node = 32
    batch_size = 64
    micro_num = batch_size // 4
    #mem_allocation = 3072

    for mem in env.MEM_SIZE:
        res = nips_partition(model, env, mem, max_node, batch_size, micro_num)
        if res is None:
            print("mem:{}  Infeasible".format(mem))
            continue
        p = translate_results(model, res, mem)
        mem_cost, t_iter = pipeline_cost(env, model, p, batch_size, micro_num,
                                         use_accumulation=False, debug=False)
        opt_result[mem] = [mem_cost, t_iter]
        print("mem:{}  cost:{}GB   iter:{}s\n".format(mem, mem_cost, t_iter))

    end_t = time.time()
    print("Total sol time:{}s".format(end_t - start_t))
