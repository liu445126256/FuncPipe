"""
This is an example for triggering a FuncPipe training program
"""
from funcpipe.platforms import Platform

# direct trigger
if __name__ == "__main__":
    partition_plan = [4, 3, 3, 3, 3, 3, 5, 1, 1]
    data_parallelism = [4, 4, 4, 4, 4, 4, 4, 4, 4]
    resource_type = [4096, 5120, 8192, 4096, 4096, 6144, 5120, 10240, 8192]

    model_name = ["amoebanet18", "amoebanet36", "resnet101", "bert-large"][-1]
    batch_size = 64

    params = {}
    # the starting rank must be 0
    params["rank"] = 0
    params["dataset_size"] = batch_size
    params["batch_size"] = batch_size
    params["micro_batchsize"] = 4
    params["epoch_num"] = 6
    params["learning_rate"] = 0.001
    params["platform"] = "aws"
    params["loss_function"] = "cross_entropy"
    params["optimizer"] = "SGD"
    params["function_name"] = "func_test"
    params["service_name"] = "funcpipe"
    params["partition_plan"] = str(partition_plan)
    params["data_parallelism"] = str(data_parallelism)
    params["resource_type"] = str(resource_type)
    params["is_init_worker"] = 1
    params["log_type"] = "file"
    params["model_name"] = model_name

    Platform.use("aws")
    Platform.invoke(params, asynchronous=True)