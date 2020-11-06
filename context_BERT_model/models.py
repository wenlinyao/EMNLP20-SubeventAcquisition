import torch
import torch.nn as nn
from torch.autograd import Variable
try:
    from pytorch_transformers import *
except:
    from transformers import *
from BERT_main import MODELS



class BasicNN(nn.Module):
    def __init__(self, args):
        super(BasicNN, self).__init__()
        self.args = args
        model_class, tokenizer_class, pretrained_weights = MODELS[0]
        
        self.bert_encoder = model_class.from_pretrained(pretrained_weights)

        if self.args.fix_embedding_layer == True:
            for name, param in self.bert_encoder.named_parameters():
                if name.startswith('embeddings'):
                    param.requires_grad = False

        #knowledge_layer_size = 20
        knowledge_layer_size = 50
        #knowledge_layer_size = 100

        self.drop = nn.Dropout(self.args.dropout)
        if self.args.knowledge_vector == True and self.args.children_vector == True:
            self.decoder = nn.Linear(self.args.bert_hidden_size*2 + knowledge_layer_size + self.args.children_vec_size, self.args.class_num)
        elif self.args.knowledge_vector == True and self.args.children_vector == False:
            self.decoder = nn.Linear(self.args.bert_hidden_size*2 + knowledge_layer_size, self.args.class_num)
        elif self.args.knowledge_vector == False and self.args.children_vector == True:
            self.decoder = nn.Linear(self.args.bert_hidden_size*2 + self.args.children_vec_size, self.args.class_num)
        else:
            self.decoder = nn.Linear(self.args.bert_hidden_size*2, self.args.class_num)

        self.knowledge_encoder = nn.Linear(self.args.knowledge_vec_size, knowledge_layer_size)
        self.knowledge_encoder2 = nn.Linear(knowledge_layer_size, knowledge_layer_size)

        self.children_vector_encoder = nn.Linear(self.args.children_vec_size*2*1, self.args.children_vec_size) # KE (multiKE) only trans vec
        #self.children_vector_encoder = nn.Linear(self.args.children_vec_size*2*3, self.args.children_vec_size) # KE (multiKE)
        #self.children_vector_encoder = nn.Linear(self.args.children_vec_size*4*1, self.args.children_vec_size) # GE
        #self.children_vector_encoder = nn.Linear(self.args.children_vec_size*4*3, self.args.children_vec_size) # multiGE
        self.children_vector_encoder2 = nn.Linear(self.args.children_vec_size, self.args.children_vec_size)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids_tensor, knowledge_vector_tensor, children_vector_tensor, masked_idx1, masked_idx2):
        # (batch_size, sequence_length, hidden_size), (batch_size, hidden_size)

        if self.args.fix_BERT == True:
            with torch.no_grad():
                self.bert_encoder.eval()
                hidden_states, pooler_output = self.bert_encoder(input_ids_tensor)
        else:
            hidden_states, pooler_output = self.bert_encoder(input_ids_tensor)
        
        batch_size, sequence_length, hidden_size = hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        masked_emb1 = torch.gather(hidden_states, dim=1, index=masked_idx1.view(batch_size,1,1).expand(batch_size,1,hidden_size))
        masked_emb2 = torch.gather(hidden_states, dim=1, index=masked_idx2.view(batch_size,1,1).expand(batch_size,1,hidden_size))

        masked_emb1 = masked_emb1.view(batch_size, hidden_size)
        masked_emb2 = masked_emb2.view(batch_size, hidden_size)

        knowledge_emb = self.knowledge_encoder(knowledge_vector_tensor)
        
        knowledge_emb = self.drop(self.ReLU(knowledge_emb))
        #knowledge_emb = self.drop(self.Tanh(knowledge_emb))
        knowledge_emb = self.knowledge_encoder2(knowledge_emb)
        

        children_vec_emb = self.children_vector_encoder(children_vector_tensor)
        children_vec_emb = self.ReLU(children_vec_emb)
        children_vec_emb = self.drop(children_vec_emb)
        #children_vec_emb = self.drop(self.ReLU(children_vec_emb))
        #children_vec_emb = self.children_vector_encoder2(children_vec_emb)
        #children_vec_emb = self.drop(self.Tanh(children_vec_emb))
        

        if self.args.knowledge_vector == True and self.args.children_vector == True:
            sentence_emb = torch.cat((masked_emb1, masked_emb2, knowledge_emb, children_vec_emb), dim=1)
        elif self.args.knowledge_vector == True and self.args.children_vector == False:
            sentence_emb = torch.cat((masked_emb1, masked_emb2, knowledge_emb), dim=1)
        elif self.args.knowledge_vector == False and self.args.children_vector == True:
            sentence_emb = torch.cat((masked_emb1, masked_emb2, children_vec_emb), dim=1)
        else:
            sentence_emb = torch.cat((masked_emb1, masked_emb2), dim=1)

        # if self.args.aggregation == "last":
        #     sentence_emb = pooler_output # (batch_size, hidden_size)
        # elif self.args.aggregation == "max":
        #     sentence_emb = torch.max(hidden_states, 1)[0]
        # elif self.args.aggregation == "mean":
        #     sentence_emb = torch.mean(hidden_states, 1)

        sentence_emb = self.drop(sentence_emb)
        
        decoded = self.decoder(sentence_emb)
        
        return decoded
        

        """
        prob = self.softmax(decoded)
        return prob
        """

