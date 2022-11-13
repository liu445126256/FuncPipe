'''This is a test of running the bert model locally'''
import sys
sys.path.append('../')

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam

from pipetests.models.bert.model import BERT, BERTLM
from pipetests.models.bert.dataset import BERTDataset, WordVocab


if __name__ == "__main__":
    # settings
    vocab_path =  "./models/bert/dataset/vocab.small"
    train_dataset_path = "./models/bert/dataset/corpus.small"
    #test_dataset_path = ""
    epoches = 1
    max_seq_len = 64
    batch_size = 4
    total_corpus_lines = 80
    on_memory = False
    bert_size = (4, 256, 4) # bert-small: layer-4 hidden-256 attn_heads-4

    print("Loading Vocab: ", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset: ", train_dataset_path)
    train_dataset = BERTDataset(train_dataset_path, vocab, seq_len=max_seq_len,
                                corpus_lines=total_corpus_lines, on_memory=on_memory)

    #print("Loading Test Dataset: ", test_dataset_path)
    #test_dataset = BERTDataset(test_dataset_path, vocab, seq_len=max_seq_len, on_memory=on_memory)

    print("Creating Dataloader")
    print("Batchsize: ", batch_size)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    #test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Building BERT model")
    vocab_size = len(vocab)
    bert = BERT(vocab_size, n_layers=bert_size[0], hidden=bert_size[1], attn_heads=bert_size[2])
    lm_model = BERTLM(bert, vocab_size)
    optimizer = Adam(lm_model.parameters(), lr=0.001, betas=(0.9, 0.999))

    print("Start training")
    for epoch_num in range(epoches):
        for i, batch in enumerate(train_data_loader):
            print("Epoch{} - batch{}".format(epoch_num, i))
            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = lm_model(batch["bert_input"], batch["segment_label"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            criteria = nn.NLLLoss(ignore_index=0)
            next_loss = criteria(next_sent_output, batch["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = criteria(mask_lm_output.transpose(1, 2), batch["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()