import torch
from torch import nn

from funcpipe.planner.profiler import Profiler
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


if __name__ == "__main__":
    Platform.use("local")
    Logger.use_logger(Logger.NATIVE, Logger.INFO, "profiler_test")

    #model = amoebanetd(num_classes=1000, num_layers=18, num_filters=256)
    #model = resnet101()
    bert_size = (8, 512, 8)  # bert-small: layer-4 hidden-256 attn_heads-4
    model = make_bert_nsp(14, n_layers=bert_size[0], hidden=bert_size[1], attn_heads=bert_size[2])

    def direct_forward_test():
        input_sample = torch.rand(1, 3, 224, 224)
        #input_sample = torch.cat([input_sample, input_sample], dim=0)
        batch_size= 3
        outputs = []
        m = get_mem_usage()
        Logger.info("start mem: {}MB".format(m))
        with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
            for bid in range(batch_size):
                out = model(input_sample)
                m = get_mem_usage()
                Logger.info("mem: {}MB".format(m))
                outputs.append(out)
        #print(prof.table())

    def cnn_model_test():
        global model
        profiler = Profiler()
        Logger.info("Starting mem usage: {}MB".format(get_mem_usage()))
        model_info = profiler.profile(model)
        del model
        activ_size = 0
        layer_size = 0
        output_size = 0
        comp_time = 0
        bp_time = 0
        grad_size = 0
        for lid in range(len(model_info.keys())):
            Logger.info("layer{:2}: size{:>10.6f}MB  activation:{:>10.6f}MB output_size:{:>10.6f}MB computation_time:{}ms    bp_time:{}ms grad_size:{:>10.6f}MB".format(lid,
                model_info[lid][0], model_info[lid][1], model_info[lid][2], model_info[lid][3],model_info[lid][4],model_info[lid][5] ))
            layer_size += model_info[lid][0]
            activ_size += model_info[lid][1]
            output_size += model_info[lid][2]
            comp_time += 0#model_info[lid][3][-1]
            bp_time += 0#model_info[lid][4][-1]
            grad_size += model_info[lid][5]
        Logger.info(
            "Total size:{:>10.6f}MB  activation:{:>10.6f}MB   output_size:{:>10.6f}MB computation_time:{:>10.6f}ms  bp_time:{:>10.6f}ms grad_size:{:>10.6f}MB".format(layer_size, activ_size,
                                                                                               output_size, comp_time, bp_time, grad_size))
        Logger.info("Ending mem usage: {:>10.6f}MB".format(get_mem_usage()))

    def bert_model_test():
        from pipetests.models.bert.dataset import BERTDataset, WordVocab
        from torch.utils.data import DataLoader
        global model
        # settings
        vocab_path = "./models/bert/dataset/vocab.small"
        train_dataset_path = "./models/bert/dataset/corpus.small"
        # test_dataset_path = ""
        dataset_size = 4
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
        profiling_loader = DataLoader(train_dataset, batch_size=1)

        profiling_sample = None
        for bid, batch in enumerate(profiling_loader):
            profiling_sample = (batch["bert_input"], batch["segment_label"])
            break

        profiler = Profiler()
        Logger.info("Starting mem usage: {}MB".format(get_mem_usage()))
        model_info = profiler.profile(model, profiling_sample)
        del model
        activ_size = 0
        layer_size = 0
        output_size = 0
        comp_time = 0
        bp_time = 0
        grad_size = 0
        for lid in range(len(model_info.keys())):
            Logger.info(
                "layer{:2}: size{:>10.6f}MB  activation:{:>10.6f}MB output_size:{:>10.6f}MB computation_time:{}ms    bp_time:{}ms grad_size:{:>10.6f}MB".format(
                    lid,
                    model_info[lid][0], model_info[lid][1], model_info[lid][2], model_info[lid][3], model_info[lid][4],
                    model_info[lid][5]))
            layer_size += model_info[lid][0]
            activ_size += model_info[lid][1]
            output_size += model_info[lid][2]
            comp_time += 0#model_info[lid][3][-1]
            bp_time += 0#model_info[lid][4][-1]
            grad_size += model_info[lid][5]
        Logger.info(
            "Total size:{:>10.6f}MB  activation:{:>10.6f}MB   output_size:{:>10.6f}MB computation_time:{:>10.6f}ms  bp_time:{:>10.6f}ms grad_size:{:>10.6f}MB".format(
                layer_size, activ_size,
                output_size, comp_time, bp_time, grad_size))
        Logger.info("Ending mem usage: {:>10.6f}MB".format(get_mem_usage()))

    #cnn_model_test()
    #direct_forward_test()
    bert_model_test()
