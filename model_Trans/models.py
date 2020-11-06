import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# https://github.com/jimmywangheng/knowledge_representation_pytorch
class TransEModel(nn.Module):
    def __init__(self, args, vocab_size, pretrained):
        super(TransEModel, self).__init__()
        self.args = args

        if args.cuda == True:
            rel_weight = torch.cuda.FloatTensor(self.args.relation_total, self.args.rnn_size*2)
        else:
            rel_weight = torch.FloatTensor(self.args.relation_total, self.args.rnn_size*2)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform(rel_weight)
        self.rel_embeddings = nn.Embedding(self.args.relation_total, self.args.rnn_size*2)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.rel_embeddings.weight.data = normalize_relation_emb

        # BiLSTM encoder
        self.vocab_size = vocab_size

        self.drop = nn.Dropout(self.args.dropout)
        self.word_encoder = nn.Embedding(self.vocab_size, self.args.embedding_size)

        self.rnn = nn.LSTM(input_size = self.args.embedding_size, hidden_size=self.args.rnn_size, num_layers = self.args.rnn_layers, batch_first=True, bidirectional=True, dropout=self.args.dropout)

        self.init_weights(pretrained = pretrained)
        print("Initialized LSTM model")

    def sentence_encoder(self, input_ids_tensor, hidden):

        words_emb = self.word_encoder(input_ids_tensor)
        output, hc = self.rnn(words_emb, hidden)
        output_last = []
        for i in range(0, len(hc[0])):
            output_last.append(hc[0][i])
        output_last = torch.cat(output_last, 1)

        if (self.args.aggregation == "mean"):
            # compress several rows to one
            output = torch.mean(output, 1)
        elif (self.args.aggregation == "last"):
            output = output_last

        elif (self.args.aggregation == "max"):
            output = torch.max(output, 1)[0]

        output = torch.squeeze(output, 1)
        return output # (batch_size, hidden_size)

    def forward(self, pos_h_tensor, pos_t_tensor, pos_r_tensor, neg_h_tensor, neg_t_tensor, neg_r_tensor, hidden):
        pos_h_e = self.sentence_encoder(pos_h_tensor, hidden)
        pos_t_e = self.sentence_encoder(pos_t_tensor, hidden)
        pos_r_e = self.rel_embeddings(pos_r_tensor)
        neg_h_e = self.sentence_encoder(neg_h_tensor, hidden)
        neg_t_e = self.sentence_encoder(neg_t_tensor, hidden)
        neg_r_e = self.rel_embeddings(neg_r_tensor)

        pos_h_e, pos_t_e, pos_r_e, neg_h_e, neg_t_e, neg_r_e = self.drop(pos_h_e), self.drop(pos_t_e), self.drop(pos_r_e), self.drop(neg_h_e), self.drop(neg_t_e), self.drop(neg_r_e)

        """
        # L1 distance
        if self.args.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        # L2 distance
        else:
        """
        pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
        neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)

        return pos, neg, [pos_h_e, pos_t_e, pos_r_e, neg_h_e, neg_t_e, neg_r_e]

    def init_weights(self, pretrained):
        """
        Initialize weights using pretrained word embedding (e.g., GloVe)
        """
        #with open("debug_log.txt", "w") as output:
        #    output.write(str(pretrained.tolist()))
        initrange = 0.1
        print("Setting pretrained embeddings")
        pretrained = pretrained.astype(np.float32)
        pretrained = torch.from_numpy(pretrained)
        if self.cuda == True:
            pretrained = pretrained.to(self.args.device)
        
        self.word_encoder.weight.data.copy_(pretrained)
        #self.word_encoder.weight.data.fill_(0)
        #self.word_encoder.weight.data.uniform_(-initrange, initrange)

        #print("word_encoder.weight.data:", self.word_encoder.weight.data)

        self.word_encoder.weight.requires_grad = self.args.trainable
        

    def init_hidden(self, batch_size):
        """
        Initialize hidden states for LSTM
        """
        #h0 = Variable(torch.zeros(self.args.rnn_layers, batch_size, self.args.rnn_size))
        #c0 = Variable(torch.zeros(self.args.rnn_layers, batch_size, self.args.rnn_size))

        # Bi-LSTM
        h0 = Variable(torch.zeros(self.args.rnn_layers * 2, batch_size, self.args.rnn_size))
        c0 = Variable(torch.zeros(self.args.rnn_layers * 2, batch_size, self.args.rnn_size))
        
        if self.args.cuda == True:
            return (h0.to(self.args.device), c0.to(self.args.device))
        else:
            return (h0, c0)
