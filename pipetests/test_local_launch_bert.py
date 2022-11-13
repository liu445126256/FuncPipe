"""
This is a debugging test for FuncPipe
Local environment used
"""
import sys
sys.path.append('../')

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from funcpipe import FuncPipe
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline

from pipetests.models.bert.model.language_model import make_bert_nsp
from pipetests.models.bert.dataset import BERTDataset, WordVocab

"""
User-defined model
"""

def test_import(params):
    print("Import and call success!")
    print(params)

# entrance function
def handler(event, context=None):
    # event = eval(event)
    Logger.use_logger(Logger.NATIVE, Logger.DEBUG, "rank%d" % int(event["rank"]))
    Logger.debug("Start")

    # training configuration
    my_rank = int(event["rank"])
    dataset_size = int(event["dataset_size"])
    batch_size = int(event["batch_size"])
    epoches = int(event["epoch_num"])
    learning_rate = float(event["learning_rate"])
    platform_type = event["platform"]

    # choose the serverless platform
    Platform.use(platform_type)

    # settings
    vocab_path = "./models/bert/dataset/vocab.small"
    train_dataset_path = "./models/bert/dataset/corpus.small"
    # test_dataset_path = ""
    max_seq_len = 64
    on_memory = False

    # loading dataset
    Logger.debug("Loading Vocab: ".format(vocab_path))
    vocab = WordVocab.load_vocab(vocab_path)
    Logger.debug("Vocab Size: ".format(len(vocab)))
    Logger.debug("Loading Train Dataset: ".format(train_dataset_path))
    train_dataset = BERTDataset(train_dataset_path, vocab, seq_len=max_seq_len,
                                corpus_lines=dataset_size, on_memory=on_memory)
    # print("Loading Test Dataset: ", test_dataset_path)
    # test_dataset = BERTDataset(test_dataset_path, vocab, seq_len=max_seq_len, on_memory=on_memory)
    Logger.debug("Creating Dataloader")
    Logger.debug("Batchsize:{}".format(batch_size))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    # test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    profiling_loader = DataLoader(train_dataset, batch_size=1)
    profiling_sample = None
    for bid, batch in enumerate(profiling_loader):
        profiling_sample = (batch["bert_input"], batch["segment_label"])
        break
    del profiling_loader

    Logger.debug("Building BERT model")
    vocab_size = len(vocab)
    bert_size = (4, 256, 4)  # bert-small: layer-4 hidden-256 attn_heads-4
    model = make_bert_nsp(vocab_size, n_layers=bert_size[0], hidden=bert_size[1], attn_heads=bert_size[2])
    optimizer = optim.Adam
    loss_func = torch.nn.NLLLoss(ignore_index=0)
    model = FuncPipe(model, loss_func=loss_func, optim_class=optimizer, learning_rate=learning_rate,
                     batch_size=batch_size)
    Logger.debug("Model built")

    # partition plan
    # we manually specify the partition for test
    # multiple stages
    def bert_partition():
        model.planner.partition_plan = [2,1,1,2]
        model.planner.tensor_parallelism = [1 for i in model.planner.partition_plan]
        model.planner.data_parallelism = [1 for i in model.planner.partition_plan]
        model.planner.micro_batchsize = 1

    bert_partition()

    # todo: some of the information can be deprecated since they have already been passed to the model
    model.init(event, profiling_sample = profiling_sample)

    #try:
    Timeline.start("Entire training process")
    for epoch_id in range(epoches):
        for batch_id, batch in enumerate(train_data_loader):
            Logger.info("Rank: %d   Epoch: %d   Batch: %d" % (my_rank, epoch_id, batch_id))
            model.pipeline_train((batch["bert_input"], batch["segment_label"]), batch["is_next"])
            #print(batch["bert_input"].shape)
            #print(batch["segment_label"].dtype)
            #exit()
    Timeline.end("Entire training process")
    #except Exception as e:
    #    Logger.info('\n\n\n' + str(e) + '\n\n')

    model.end()

# direct trigger
if __name__ == "__main__":
    params = {}
    # the starting rank must be 0
    params["rank"] = 0
    params["dataset_size"] = 4
    params["batch_size"] = 4
    params["epoch_num"] = 1
    params["learning_rate"] = 0.001
    params["platform"] = "local"
    params["function_name"] = "pipetests.test_local_launch_bert.handler"

    handler(params)