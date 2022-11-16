from __future__ import print_function
import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from utils.plot import plot
from utils.tokenize import tokenize, create_dict, sent_to_ix, pad_feature
from torch.utils.data import Dataset

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Iemocap_Dataset(Dataset):
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(Iemocap_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        self.dataroot = os.path.join(dataroot, 'IEMOCAP')
        self.private_set = name == 'private'

        # if name == 'train':
        #     name = 'traindev'
        # if name == 'valid':
        #     name = 'test'

        word_file = os.path.join(self.dataroot, name + "_sentences.p")
        audio_file = os.path.join(self.dataroot, name + "_mels.p")
        y_s_file = os.path.join(self.dataroot, name + "_emotions.p")

        self.key_to_word = pickle.load(open(word_file, "rb"))
        self.key_to_audio = pickle.load(open(audio_file, "rb"))
        self.key_to_label = pickle.load(open(y_s_file, "rb"))
        self.set = list(self.key_to_label.keys())

        for key in self.set:
            if not (key in self.key_to_word and
                    key in self.key_to_audio and
                    key in self.key_to_label):
                print("Not present everywhere, removing key ", key)
                self.set.remove(key)

        # Plot temporal dimension of feature
        # t = []
        # for key in self.key_to_word.keys():
        #     x = np.array(self.key_to_word[key]).shape[0]
        #     t.append(x)
        # plot(t)
        # sys.exit()

        # Creating embeddings and word indexes
        self.key_to_sentence = tokenize(self.key_to_word)
        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = create_dict(self.key_to_sentence, self.dataroot)
        self.vocab_size = len(self.token_to_ix)

        self.l_max_len = 15
        self.a_max_len = 40

    def __getitem__(self, idx):
        key = self.set[idx]
        L = sent_to_ix(self.key_to_sentence[key], self.token_to_ix, max_token=self.l_max_len)
        A = pad_feature(self.key_to_audio[key], self.a_max_len)
        V = np.zeros(1) # not using video, insert dummy

        y = self.key_to_label[key]
        y = np.array(y)

        if self.args.model == "Model_MISA":
            ## BERT-based features input prep
            SENT_LEN = len(L)
            # Create bert indices using tokenizer
            bert_details = []
            W = self.key_to_word[key]
            text = " ".join(W)
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN + 2, add_special_tokens=True, pad_to_max_length=True)
            # Bert things are batch_first
            bert_sentences = torch.LongTensor(encoded_bert_sent["input_ids"])
            bert_sentence_types = torch.LongTensor(encoded_bert_sent["token_type_ids"])
            bert_sentence_att_mask = torch.LongTensor(encoded_bert_sent["attention_mask"])
            # lengths are useful later in using RNNs
            lengths = torch.zeros(L.shape[0], dtype=torch.int64)
            return torch.from_numpy(L), torch.from_numpy(V).float(), torch.from_numpy(A), torch.from_numpy(y), lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask
        return key, torch.from_numpy(L), torch.from_numpy(A), torch.from_numpy(V).float(), torch.from_numpy(y)

    def __len__(self):
        return len(self.set)
