import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_transformers import *
from BERT_context_model_main import MODELS
import numpy as np



class BERT_NN(nn.Module):
    def __init__(self, args):
        super(BERT_NN, self).__init__()
        self.args = args
        model_class, tokenizer_class, pretrained_weights = MODELS[0]
        
        self.bert_encoder = model_class.from_pretrained(pretrained_weights)

        if self.args.fix_embedding_layer == True:
            for name, param in self.bert_encoder.named_parameters():
                if name.startswith('embeddings'):
                    param.requires_grad = False

        self.drop = nn.Dropout(self.args.dropout)

        self.decoder = nn.Linear(self.args.bert_hidden_size*2, self.args.class_num)

        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids_tensor, masked_idx1, masked_idx2):
        # (batch_size, sequence_length, hidden_size), (batch_size, hidden_size)

        hidden_states, pooler_output = self.bert_encoder(input_ids_tensor)
        
        batch_size, sequence_length, hidden_size = hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        masked_emb1 = torch.gather(hidden_states, dim=1, index=masked_idx1.view(batch_size,1,1).expand(batch_size,1,hidden_size))
        masked_emb2 = torch.gather(hidden_states, dim=1, index=masked_idx2.view(batch_size,1,1).expand(batch_size,1,hidden_size))

        masked_emb1 = masked_emb1.view(batch_size, hidden_size)
        masked_emb2 = masked_emb2.view(batch_size, hidden_size)

        sentence_emb = torch.cat((masked_emb1, masked_emb2), dim=1)

        # if self.args.aggregation == "last":
        #     sentence_emb = pooler_output # (batch_size, hidden_size)
        # elif self.args.aggregation == "max":
        #     sentence_emb = torch.max(hidden_states, 1)[0]
        # elif self.args.aggregation == "mean":
        #     sentence_emb = torch.mean(hidden_states, 1)

        sentence_emb = self.drop(sentence_emb)
        
        decoded = self.decoder(sentence_emb)
        prob = self.softmax(decoded)

        return prob