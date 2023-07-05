try:
    import unzip_requirements
except ImportError:
    print("Import unzip failed!")

import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle

from funcpipe.planner.new_profiler import Profiler
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.utils import get_mem_usage

"""
User-defined model
"""
from pipetests.models.resnet import resnet101
from pipetests.models.amoebanet import amoebanetd
from pipetests.models.bert.model.language_model import make_bert_nsp
from pipetests.models.bert.dataset import BERTDataset, WordVocab


# entrance function
def handler(event, context=None):
    Platform.use("aws")
    Logger.use_logger(Logger.HTTP, Logger.DEBUG, "new_profiler_test")

    # model = resnet101()  # amoebanetd(num_classes=1000, num_layers=18, num_filters=256)
    # model = amoebanetd(num_classes=1000, num_layers=36, num_filters=256)

    model_name = "bert-large"
    profiling_sample = None

    if model_name == "resnet101":
        model = resnet101()
    elif model_name == "amoebanet18":
        model = amoebanetd(num_classes=1000, num_layers=18, num_filters=256)
    elif model_name == "amoebanet36":
        model = amoebanetd(num_classes=1000, num_layers=36, num_filters=256)
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
        dataset_size = 4
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
        profiling_loader = DataLoader(train_dataset, batch_size=1)
        for bid, batch in enumerate(profiling_loader):
            profiling_sample = (batch["bert_input"], batch["segment_label"])
            break
        Logger.debug("Building BERT model")
        vocab_size = len(vocab)
        Logger.debug("Before making nsp")
        model = make_bert_nsp(vocab_size, n_layers=bert_size[0], hidden=bert_size[1], attn_heads=bert_size[2])
        Logger.debug("After making nsp")

    else:
        Logger.debug("Invalid model!")
        raise Exception("Invalid model!")

    profiler = Profiler()
    Logger.info("Starting mem usage: {}MB".format(get_mem_usage()))
    res = profiler.profile(model,  [1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240], batch_size=4, input_sample=profiling_sample)
    Logger.info(str(res))
    # upload to storage
    res_data = pickle.dumps(res)
    Platform.upload_to_storage("prof_bert_large_8level", res_data)

    return {}

