"""
This is an example for triggering a FuncPipe training program
"""
from funcpipe.platforms import Platform
from funcpipe.configs import Config

# direct trigger
if __name__ == "__main__":
    # partition and resource configuration policy
    partition_plan = [39, 95, 81, 89]
    data_parallelism = [1, 1, 1, 1]
    resource_type = [3072, 3072, 3072, 3072]
    model_name = ["amoebanet18", "amoebanet36", "resnet101", "bert-large"][2]
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
    params["function_name"] = Config.getvalue("platform-aws", "trainer_function_name")
    params["service_name"] = "funcpipe"
    params["partition_plan"] = str(partition_plan)
    params["data_parallelism"] = str(data_parallelism)
    params["resource_type"] = str(resource_type)
    params["is_init_worker"] = 1
    params["log_type"] = "http"
    params["model_name"] = model_name

    Platform.use("aws")
    Platform.invoke(params, asynchronous=True)