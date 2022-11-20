"""
This is a debugging test for FuncPipe in the cloud environment
"""
try:
    import unzip_requirements
except ImportError:
    print("Import unzip failed!")

import sys
# sys.path.append('../')
import traceback

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from funcpipe import FuncPipe
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline

"""
User-defined model
"""
from pipetests.models.resnet import resnet101
from pipetests.models.amoebanet import amoebanetd
from pipetests.models.bert.model.language_model import make_bert_nsp
from pipetests.models.bert.dataset import BERTDataset, WordVocab


# entrance function
def handler(event, context=None):
    # input training configuration
    my_rank = int(event["rank"])
    dataset_size = int(event["dataset_size"])
    batch_size = int(event["batch_size"])
    epoches = int(event["epoch_num"])
    learning_rate = float(event["learning_rate"])
    platform_type = event["platform"]
    loss_func = getattr(F, event["loss_function"])
    optimizer = getattr(optim, event["optimizer"])
    partition_plan = eval(event["partition_plan"])
    data_parallelism = eval(event["data_parallelism"])
    resource_type = eval(event["resource_type"])
    log_type = event["log_type"]
    model_name = event["model_name"]
    micro_batchsize = int(event["micro_batchsize"])

    if log_type == "file":
        Logger.use_logger(Logger.FILE, Logger.DEBUG, "rank%d" % int(event["rank"]))
    elif log_type == "http":
        Logger.use_logger(Logger.HTTP, Logger.DEBUG, "rank%d" % int(event["rank"]))
    else:
        Logger.use_logger(Logger.HTTP, Logger.DEBUG, "rank%d" % int(event["rank"]))
        Logger.debug(log_type)
    #Logger.use_logger(Logger.HTTP, Logger.DEBUG, "Init worker")
    Logger.debug("Start")
    iter_num_per_epoch = dataset_size // batch_size

    # choose the serverless platform
    Platform.use(platform_type)

    ##############################################
    # user-defined model (sequential model required)
    if model_name == "amoebanet18":
        model = amoebanetd(num_classes=1000, num_layers=18, num_filters=256)  # amoebanetd(num_classes=1000, num_layers=9, num_filters=64)
    elif model_name == "amoebanet36":
        model = amoebanetd(num_classes=1000, num_layers=36, num_filters=256)
    elif model_name == "resnet101":
        model = resnet101()
    elif "bert" in model_name:
        if model_name == "bert-base":
            bert_size = (12, 768, 12)
        elif model_name == "bert-large":
            bert_size = (24, 1024, 16)  # bert-small: layer-4 hidden-256 attn_heads-4
        else:
            Logger.debug("Invalid bert model!")
            raise Exception("Invalid bert model!")
        # settings
        vocab_path = "./pipetests/models/bert/dataset/vocab.small"
        train_dataset_path = "./pipetests/models/bert/dataset/corpus.small"
        max_seq_len = 64
        on_memory = False
        # loading dataset
        Logger.debug("Loading Vocab: ".format(vocab_path))
        with open(train_dataset_path, "r", encoding="utf-8") as f:
            vocab = WordVocab(f, max_size=None, min_freq=1)
        Logger.debug("Vocab Size:{} ".format(len(vocab)))
        Logger.debug("Loading Train Dataset:{} ".format(train_dataset_path))
        train_dataset = BERTDataset(train_dataset_path, vocab, seq_len=max_seq_len,
                                    corpus_lines=dataset_size, on_memory=on_memory)
        Logger.debug("Creating Dataloader")
        Logger.debug("Batchsize:{}".format(batch_size))
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
        # test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

        Logger.debug("Building BERT model")
        vocab_size = len(vocab)
        Logger.debug("Before making nsp")
        model = make_bert_nsp(vocab_size, n_layers=bert_size[0], hidden=bert_size[1], attn_heads=bert_size[2])
        Logger.debug("After making nsp")
    else:
        Logger.debug("Invalid model type!")
        raise Exception("Invalid model type!")
    ##############################################

    # wrap the user-defined model
    Logger.debug("Model built")
    model = FuncPipe(model, loss_func=loss_func, optim_class=optimizer, learning_rate=learning_rate,
                     batch_size=batch_size)

    # resource configuration
    model.planner.partition_plan = partition_plan
    model.planner.tensor_parallelism = [1 for i in range(len(partition_plan))]
    model.planner.data_parallelism = data_parallelism
    model.planner.mem_allocation = resource_type
    model.planner.micro_batchsize = micro_batchsize

    # init pipeline
    model.init(event)

    # start training
    try:
        Timeline.start("Entire training process-")
        if "bert" not in model_name:
            for epoch_id in range(epoches):
                for batch_id in range(iter_num_per_epoch):
                    Logger.info("Rank: %d   Epoch: %d   Batch: %d" % (my_rank, epoch_id, batch_id))
                    model.pipeline_train(None, None)
        else:
            for epoch_id in range(epoches):
                for batch_id, batch in enumerate(train_data_loader):
                    Logger.info("Rank: %d   Epoch: %d   Batch: %d" % (my_rank, epoch_id, batch_id))
                    model.pipeline_train((batch["bert_input"], batch["segment_label"]), batch["is_next"])
        Timeline.end("Entire training process-")

    except Exception as e:
        str_error = str(traceback.format_exc())
        Logger.info('\n\n\n' + str(str_error) + '\n\n\n')
        return {}
    model.end()

    return {}