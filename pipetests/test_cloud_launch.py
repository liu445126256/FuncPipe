"""
This is a debugging test for FuncPipe in the cloud environment
"""
import sys
#sys.path.append('../')

import torch
import torch.optim as optim
import torch.nn.functional as F

from funcpipe import FuncPipe
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline

"""
User-defined model
"""
from pipetests.models.resnet import resnet101

def test_import(params):
    print("Import and call success!")
    print(params)

# entrance function
def handler(event, context=None):
    #event = eval(event)

    Logger.use_logger(Logger.HTTP, Logger.DEBUG, "rank%d" % int(event["rank"]))
    Logger.debug("Start")

    # training configuration
    my_rank = int(event["rank"])
    dataset_size = int(event["dataset_size"])
    batch_size = int(event["batch_size"])
    epoches = int(event["epoch_num"])
    learning_rate = float(event["learning_rate"])
    platform_type = event["platform"]
    # function_name = event["function_name"] # set but only used by func_manager
    loss_func = getattr(F, event["loss_function"])
    optimizer = getattr(optim, event["optimizer"])

    # generate input data
    input = torch.rand(batch_size, 3, 224, 224)
    target = torch.randint(1000, (batch_size,)) # num classes = 1000
    data = [(input, target)] * (dataset_size // batch_size)
    if dataset_size % batch_size != 0:
        last_input = input[:dataset_size % batch_size]
        last_target = target[:dataset_size % batch_size]
        data.append((last_input, last_target))

    # choose the serverless platform
    Platform.use(platform_type)

    # FuncPipe training
    model = resnet101()  # num classes = 1000
    Logger.debug("Model built")
    model = FuncPipe(model, loss_func = loss_func, optim_class = optimizer, learning_rate = learning_rate, batch_size = batch_size)

    # partition plan
    # we manually specify the partition for test
    # multiple stages
    def single_stage():
        model.planner.partition_plan = [304]
        model.planner.tensor_parallelism = [1]
        model.planner.data_parallelism = [1]
        model.planner.micro_batchsize = 1
    def two_stages():
        model.planner.partition_plan = [67, 237] # layer2 | layer3
        model.planner.tensor_parallelism = [1, 1]
        model.planner.data_parallelism = [1, 1]
        model.planner.micro_batchsize = 1
    def four_stages():
        model.planner.partition_plan = [31, 36, 107, 130] # layer1 | layer2 | layer3 | layer4
        model.planner.tensor_parallelism = [1, 1, 1, 1]
        model.planner.data_parallelism = [1, 1, 1, 1]
        model.planner.micro_batchsize = 1
    def two_stage_data_parallel_balanced():
        model.planner.partition_plan = [67, 237]  # layer2 | layer3
        model.planner.tensor_parallelism = [1, 1]
        model.planner.data_parallelism = [2, 2]
        model.planner.micro_batchsize = 1
    def two_stage_data_parallel_imbalanced():
        model.planner.partition_plan = [67, 237]  # layer2 | layer3
        model.planner.tensor_parallelism = [1, 1]
        model.planner.data_parallelism = [4, 2]
        model.planner.micro_batchsize = 1
    def four_stage_data_parallel_imbalanced():
        model.planner.partition_plan = [31, 36, 107, 130] # layer1 | layer2 | layer3 | layer4
        model.planner.tensor_parallelism = [1, 1, 1, 1]
        model.planner.data_parallelism = [2, 1, 2, 2]
        model.planner.micro_batchsize = 1
    def data_parallel():
        model.planner.partition_plan = [304]
        model.planner.tensor_parallelism = [1]
        model.planner.data_parallelism = [4]
        model.planner.micro_batchsize = 4

    #two_stage_data_parallel_imbalanced()
    #four_stages()
    data_parallel()

    # todo: some of the information can be deprecated since they have already been passed to the model
    model.init(event)

    Timeline.start("Entire training process")
    for epoch_id in range(epoches):
        for batch_id, (inputs, targets) in enumerate(data):
            Logger.info("Rank: %d   Epoch: %d   Batch: %d" % (my_rank, epoch_id, batch_id))
            model.pipeline_train(inputs, targets)
    Timeline.end("Entire training process")

    model.end()

# parameters for function triggering call
if __name__ == "__main__":
    params = {}
    # the starting rank must be 0
    params["rank"] = 0
    params["dataset_size"] = 32
    params["batch_size"] = 16
    params["epoch_num"] = 1
    params["learning_rate"] = 0.001
    params["platform"] = "local"
    params["loss_function"] = "cross_entropy"
    params["optimizer"] = "SGD"

    params["function_name"] = "funcpipe-test-0"
    params["service_name"] = "funcpipe"

    handler(params)