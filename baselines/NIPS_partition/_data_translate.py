import json



#############################################
# Our data structure
#############################################
class ali_env:
    BW = 50 # MB/s, bandwidth to oss storage
    LAT = 50 / 1000.0 # s, latency to oss storage
    MEM_LIMIT = 3096 # MB
    UNIT_COST = 0.000110592 # GB.s

    MEM_OPTION_LEVEL = 4
    MEM_SIZE = [256, 512, 1024, 3072]
    COMP_RATIO = [0.1, 0.2, 0.4, 1]
    COMM_RATIO = [1, 1, 1, 1]
    def __init__(self):
        pass

class dnn_model:
    MAX_LAYER = 10
    def __init__(self, model_info, base_mem ,max_layer_num = 9999):
        self.layer_num = -1
        self.layer_size = []
        self.layer_act_size = []
        self.layer_fwd_time = []  # (alpha, beta)
        self.layer_bp_time = []  # (alpha, beta)
        self.layer_output_size = []
        self.layer_grad_size = []
        self.base_mem = base_mem  # MB basic mem cost
        self.max_layer_num = max_layer_num#self.MAX_LAYER

        self._load_model_info(model_info)

        if self.layer_num > self.max_layer_num:
            self._merge_layer()
        #exit()


    def _load_model_info(self, model_info):
        '''
        :param model_info: (layer size, act size, output size, fwd time, bp time, grad size)
        :return:
        '''
        self.layer_num = len(model_info)#min(len(model_info), self.max_layer_num)
        for i in range(self.layer_num):
            layer_info = model_info[i]
            self.layer_size.append(layer_info[0])
            self.layer_act_size.append(layer_info[1])
            self.layer_output_size.append(layer_info[2])
            self.layer_fwd_time.append(
                (layer_info[3][0] / 1000.0, layer_info[3][1] / 1000.0))  # ((0, layer_info[3] / 1000.0)) #
            self.layer_bp_time.append(
                (layer_info[4][0] / 1000.0, layer_info[4][1] / 1000.0))  # ((0, layer_info[4] / 1000.0)) #
            # self.layer_fwd_time.append( (0, layer_info[3] / 1000.0))
            # self.layer_bp_time.append( (0, layer_info[4] / 1000.0))
            self.layer_grad_size.append(layer_info[5])

    def _merge_layer(self):
        max_layer = self.max_layer_num

        compress_ratio = self.layer_num // max_layer + 1
        compressed_layer_num = [compress_ratio for i in range(self.max_layer_num)]
        diff = max_layer * compress_ratio - self.layer_num
        for i in range(self.max_layer_num - 1, self.max_layer_num - 1 - diff, -1):
            compressed_layer_num[i] = compressed_layer_num[i] - 1

        layer_num = 0
        layer_size = []
        layer_act_size = []
        layer_fwd_time = []  # (alpha, beta)
        layer_bp_time = []  # (alpha, beta)
        layer_output_size = []
        layer_grad_size = []

        ls = 0
        las = 0
        lft = [0, 0]
        lbt = [0, 0]
        los = 0
        lgs = self.layer_grad_size[0]
        new_layer_id = 0
        current_layers = 0
        for i in range(self.layer_num):
            assert lgs >= 0
            current_layers += 1
            ls += self.layer_size[i]
            las += self.layer_act_size[i]
            lft[0] += self.layer_fwd_time[i][0]
            lbt[0] += self.layer_bp_time[i][0]
            lft[1] += self.layer_fwd_time[i][1]
            lbt[1] += self.layer_bp_time[i][1]
            los = self.layer_output_size[i]  # gradient and output to send does not accumulate
            #lgs = self.layer_grad_size[i]
            if current_layers == compressed_layer_num[new_layer_id]:  # (i + 1) % compress_ratio == 0:
                new_layer_id += 1
                current_layers = 0
                layer_size.append(ls)
                layer_act_size.append(las)
                layer_fwd_time.append(lft)
                layer_bp_time.append(lbt)
                layer_output_size.append(los)
                layer_grad_size.append(lgs)
                ls = 0
                las = 0
                lft = [0, 0]
                lbt = [0, 0]
                los = 0
                if i == self.layer_num - 1: lgs = -1
                else: lgs = self.layer_grad_size[i + 1]
        print(compressed_layer_num, self.layer_num)
        #exit()
        assert ls == 0
        assert sum(compressed_layer_num) == self.layer_num
        assert new_layer_id == max_layer

        self.layer_num = max_layer#layer_num
        self.layer_size = layer_size
        self.layer_act_size = layer_act_size
        self.layer_fwd_time = layer_fwd_time
        self.layer_bp_time = layer_bp_time
        self.layer_output_size = layer_output_size
        self.layer_grad_size = layer_grad_size

        print("compressed layer num:{}".format(compressed_layer_num))




def format_disp(data_name):
    with open(data_name, "r") as f:
        graph = json.load(f)
    print(graph.keys())
    print(graph)



def translate_model_info(model: dnn_model, env, mem_size, max_node_num, batch_size, micro_batch_num):
    micro_batch_size = batch_size // micro_batch_num

    t_fwd = []
    t_bp = []
    t_comm_fw_send = []
    t_comm_fw_recv = []
    t_comm_bp_send = []
    t_comm_bp_recv = []
    mem_cost = []

    resource_id = env.MEM_SIZE.index(mem_size)
    '''
    comp_ratio = []  # env.COMP_RATIO[resource_id]
    comm_ratio = []  # env.COMM_RATIO[resource_id]
    for i in range(model.layer_num):
        # comp/comm capability
        comp_ratio.append(env.COMP_RATIO[resource_id])
        comm_ratio.append(env.COMM_RATIO[resource_id])
    '''
    fwd_ratio = []  # env.COMP_RATIO[resource_id]
    bp_ratio = []
    comm_ratio = []  # env.COMM_RATIO[resource_id]
    for i in range(model.layer_num):
        # comp/comm capability
        fwd_ratio.append(env.FWD_RATIO[resource_id])
        bp_ratio.append(env.BP_RATIO[resource_id])
        comm_ratio.append(env.COMM_RATIO[resource_id])


    for i in range(model.layer_num):
        # stage time
        fwd_time = (model.layer_fwd_time[i][0] + model.layer_fwd_time[i][1] * micro_batch_size) / fwd_ratio[i]  # / plc.data_parallel[i]
        t_fwd.append(fwd_time)
        bp_time = (model.layer_bp_time[i][0] + model.layer_bp_time[i][1] * micro_batch_size) / bp_ratio[i]  # / plc.data_parallel[i]
        t_bp.append(bp_time)
        if i == model.layer_num - 1:
            fwd_send_time = fwd_recv_time = 0
        else:
            fwd_send_time = (model.layer_output_size[i] * micro_batch_size / env.BW + env.LAT) / \
                            comm_ratio[i]  # / plc.data_parallel[i]
            fwd_recv_time = (model.layer_output_size[i] * micro_batch_size / env.BW + env.LAT)  / \
                            comm_ratio[i + 1]  # / plc.data_parallel[i]
        t_comm_fw_send.append(fwd_send_time)
        t_comm_fw_recv.append(fwd_recv_time)
        if i == 0:
            bp_send_time = bp_recv_time = 0
        else:
            bp_send_time = (model.layer_grad_size[i] * micro_batch_size / env.BW + env.LAT) / \
                           comm_ratio[i]  # / plc.data_parallel[i]
            bp_recv_time = (model.layer_grad_size[i] * micro_batch_size / env.BW + env.LAT) / \
                           comm_ratio[i - 1]  # / plc.data_parallel[i]

        t_comm_bp_send.append(bp_send_time)
        t_comm_bp_recv.append(bp_recv_time)

        mem_cost.append(model.layer_act_size[i] * batch_size + model.layer_size[i])

    graph = {}
    graph['maxSizePerFPGA'] = (mem_size - model.base_mem - 200) * 1024 * 1024 # extra 200 to avoid OOM
    graph['maxFPGAs'] = max_node_num
    graph['maxCPUs'] = 0
    graph['nodes'] = []
    graph['edges'] = []

    for i in range(model.layer_num):
        fwd_id = i
        bp_id = i + model.layer_num

        # fwd node
        new_node = {}
        new_node["name"] = "layer{}_fwd".format(fwd_id)
        new_node["id"] = fwd_id
        new_node["supportedOnFpga"] = 1
        new_node["cpuLatency"] = 0
        new_node["fpgaLatency"] = t_fwd[i]
        new_node["isBackwardNode"] = 0
        new_node["colorClass"] = i
        new_node['size'] = mem_cost[i] * 1024 * 1024
        #print(mem_cost)
        #print(model.layer_act_size)
        #exit()
        graph['nodes'].append(new_node)

        # bp node
        new_node = {}
        new_node["name"] = "layer{}_bp".format(bp_id)
        new_node["id"] = bp_id
        new_node["supportedOnFpga"] = 1
        new_node["cpuLatency"] = 0
        new_node["fpgaLatency"] = t_bp[i]
        new_node["isBackwardNode"] = 1
        new_node["colorClass"] = i
        new_node['size'] = 0
        graph['nodes'].append(new_node)

        # edge
        fwd_egde = {}
        fwd_egde['sourceId'] = fwd_id
        if i < model.layer_num - 1:
            fwd_egde['destId'] = i + 1
            fwd_egde["cost"] = t_comm_fw_send[i] + t_comm_fw_recv[i]
        else:
            fwd_egde['destId'] = bp_id
            fwd_egde["cost"] = 0
        graph['edges'].append(fwd_egde)

        if i > 0:
            bp_egde = {}
            #bp_egde['sourceId'] = bp_id - 1
            #bp_egde['destId'] = bp_id
            bp_egde['sourceId'] = bp_id
            bp_egde['destId'] = bp_id - 1
            bp_egde["cost"] = t_comm_bp_send[i] + t_comm_bp_recv[i]
            graph['edges'].append(bp_egde)

    print(graph)

    return graph




if __name__ == "__main__":
    #format_disp("./bert24_training.json")
    #MEM_SIZE = [256, 512, 1024, 3072]

    import pickle
    f = open("../../tools/resnet_101", "rb")
    model_info, base_mem = pickle.load(f)
    model = dnn_model(model_info, base_mem)
    translate_model_info(model, ali_env, 3072, 32, 256, 64)

