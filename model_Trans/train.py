import sys, argparse, random, tqdm, pickle, time, glob
sys.path.append("../utilities/")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from pytorch_transformers import *
from models import *
from utilities import EventPair, remove_brackets

def tensor_to_numpy(x):
    ''' Need to cast before calling numpy()
    '''
    #return (Variable(x).data).cpu().numpy()
    return x.data.type(torch.DoubleTensor).numpy()

def distance(h_emb, t_emb, rel_emb):
    return sum((np.array(h_emb) - np.array(rel_emb) - np.array(t_emb))**2)

class marginLoss(nn.Module):
    def __init__(self, args):
        super(marginLoss, self).__init__()
        self.args = args

    def forward(self, pos, neg):
        if self.args.cuda == True:
            zero_tensor = torch.cuda.FloatTensor(pos.size())
        else:
            zero_tensor = torch.FloatTensor(pos.size())
        zero_tensor.zero_()
        zero_tensor = torch.autograd.Variable(zero_tensor)
        return torch.sum(torch.max(pos - neg, zero_tensor))

class Experiment:
    def __init__(self, args, tokenizer):
        self.args = args
        
        self.env = pickle.load(open("env.pkl", "rb"))

        self.train_set = self.env['train']
        self.dev_set = self.env['dev']
        self.test_set = self.env['test']

        if(self.args.toy == True):
            print("Using toy mode...")
            random.shuffle(self.train_set)
            self.train_set = self.train_set[:2000]
            self.dev_set = self.dev_set[:500]
            random.shuffle(self.test_set)
            self.test_set = self.test_set[:500]

        """
        if args.experiment == "BERT":
            self.mdl = BERT_NN(args)
        elif args.experiment == "LSTM":
        """
        self.mdl = TransEModel(args, len(self.env["word_index"]), pretrained=self.env["glove"])

        if self.args.cuda == True:
            self.mdl.to(self.args.device)

    def select_optimizer(self):
        parameters = filter(lambda p: p.requires_grad, self.mdl.parameters())
        self.optimizer =  optim.Adam(parameters, lr=self.args.learning_rate)
    
    # def select_optimizer(self):
    #     no_decay = ['bias', 'LayerNorm.weight']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in self.mdl.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
    #         {'params': [p for n, p in self.mdl.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #         ]
    #     self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    def make_batch(self, x, i):
        ''' 
        :param x: input sentences
        :param i: select the ith batch (-1 to take all)
        :return: sentences, targets, actual_batch_size
        '''
        batch = x[int(i * self.args.batch_size):int((i + 1) * self.args.batch_size)]

        pos_head_ids_tensor = torch.tensor([instance['pos_head_ids'] for instance in batch])
        pos_tail_ids_tensor = torch.tensor([instance['pos_tail_ids'] for instance in batch])
        pos_rel_tensor = torch.tensor([instance['pos_rel'] for instance in batch])
        neg_head_ids_tensor = torch.tensor([instance['neg_head_ids'] for instance in batch])
        neg_tail_ids_tensor = torch.tensor([instance['neg_tail_ids'] for instance in batch])
        neg_rel_tensor = torch.tensor([instance['neg_rel'] for instance in batch])
        
        
        """
        if self.args.experiment == "BERT":
            masked_idx1 = torch.LongTensor(np.array([instance['masked_idxList'][0] for instance in batch], dtype=np.int32).tolist())
            masked_idx2 = torch.LongTensor(np.array([instance['masked_idxList'][1] for instance in batch], dtype=np.int32).tolist())
        else:
            masked_idx1, masked_idx2 = None, None
        """

        if self.args.cuda == True:
            pos_head_ids_tensor = pos_head_ids_tensor.to(self.args.device)
            pos_tail_ids_tensor = pos_tail_ids_tensor.to(self.args.device)
            pos_rel_tensor = pos_rel_tensor.to(self.args.device)
            neg_head_ids_tensor = neg_head_ids_tensor.to(self.args.device)
            neg_tail_ids_tensor = neg_tail_ids_tensor.to(self.args.device)
            neg_rel_tensor = neg_rel_tensor.to(self.args.device)
            
            """
            if self.args.experiment == "BERT":
                masked_idx1 = masked_idx1.to(self.args.device)
                masked_idx2 = masked_idx2.to(self.args.device)
            """

        actual_batch_size = pos_head_ids_tensor.size(0)
        pos_head_ids_tensor = Variable(pos_head_ids_tensor, requires_grad=False)
        pos_tail_ids_tensor = Variable(pos_tail_ids_tensor, requires_grad=False)
        pos_rel_tensor = Variable(pos_rel_tensor, requires_grad=False)
        neg_head_ids_tensor = Variable(neg_head_ids_tensor, requires_grad=False)
        neg_tail_ids_tensor = Variable(neg_tail_ids_tensor, requires_grad=False)
        neg_rel_tensor = Variable(neg_rel_tensor, requires_grad=False)
        

        return pos_head_ids_tensor, pos_tail_ids_tensor, pos_rel_tensor, neg_head_ids_tensor, neg_tail_ids_tensor, neg_rel_tensor, actual_batch_size

    def train_batch(self, i):
        self.mdl.train()
        pos_head_ids_tensor, pos_tail_ids_tensor, pos_rel_tensor, neg_head_ids_tensor, neg_tail_ids_tensor, neg_rel_tensor, actual_batch_size = self.make_batch(self.train_set, i)

        #self.mdl.zero_grad()
        self.optimizer.zero_grad()

        """
        if self.args.experiment == "BERT":
            output = self.mdl(input_ids_tensor, masked_idx1, masked_idx2)
        elif self.args.experiment == "LSTM":
        """
        hidden = self.mdl.init_hidden(actual_batch_size)
        pos, neg, _ = self.mdl(pos_head_ids_tensor, pos_tail_ids_tensor, pos_rel_tensor, neg_head_ids_tensor, neg_tail_ids_tensor, neg_rel_tensor, hidden)
        
        loss = self.criterion(pos, neg)
        
        loss.backward()

        nn.utils.clip_grad_norm_(parameters = self.mdl.parameters(), max_norm = self.args.max_grad_norm)
        self.optimizer.step()

        #return loss.data[0]
        return loss.item()

    def train(self):
        """
        This is the main train function
        """
        self.criterion = marginLoss(self.args)
        print(self.args)
        total_loss = 0

        if len(self.train_set) % self.args.batch_size == 0:
            num_batches = int(len(self.train_set) / self.args.batch_size)
        else:
            num_batches = int(len(self.train_set) / self.args.batch_size) + 1

        print("len(self.train_set)", len(self.train_set))
        print("num_batches:", num_batches)
        self.select_optimizer()

        best_acc = 0
        best_F = 0
        final_results_strList = []
        for epoch in range(1, self.args.epochs+1):
            self.mdl.train()
            print("epoch: ", epoch)
            t0 = time.clock()
            random.shuffle(self.train_set)
            print("========================================================================")
            losses = []
            for i in tqdm(range(num_batches)):
                loss = self.train_batch(i)
                if(loss is None):
                    continue    
                losses.append(loss)
            t1 = time.clock()
            print("[Epoch {}] Train Loss={} T={}s".format(epoch, np.mean(losses),t1-t0))

            if len(self.dev_set) != 0 and epoch % 5 == 0:
                print("Evaluate on dev set...")
                acc = self.test(epoch, "dev")
                print("acc:", acc)

                self.test(epoch, "test")
    
    def test(self, epoch, data_flag):
        if data_flag == "dev":
            dataset = self.dev_set
            output_file = open("dev_emb_" + str(epoch) + ".txt", "w")

        elif data_flag == "test":
            dataset = self.test_set
            output_file = open("test_emb_" + str(epoch) + ".txt", "w")

        head_embList, tail_embList, rel_embList, acc = self.evaluate(dataset)

        event2vec = {}
        for i, instance in enumerate(dataset):
            eventpair = EventPair(instance["pos_pair"], -1)
            if eventpair.event1 not in event2vec:
                event2vec[eventpair.event1] = head_embList[i]
            if eventpair.event2 not in event2vec:
                event2vec[eventpair.event2] = tail_embList[i]
        for event in event2vec:
            output_file.write(event + "\t" + str(event2vec[event]) + "\n")
        output_file.close()

        return acc
        

    def evaluate(self, x):
        self.mdl.eval()

        if len(x) % self.args.batch_size == 0:
            num_batches = int(len(x) / self.args.batch_size)
        else:
            num_batches = int(len(x) / self.args.batch_size) + 1

        pos_head_embList = []
        pos_tail_embList = []
        pos_rel_embList = []

        neg_head_embList = []
        neg_tail_embList = []
        neg_rel_embList = []

        for i in range(num_batches):
            pos_head_ids_tensor, pos_tail_ids_tensor, pos_rel_tensor, neg_head_ids_tensor, neg_tail_ids_tensor, neg_rel_tensor, actual_batch_size = self.make_batch(x, i)
            hidden = self.mdl.init_hidden(actual_batch_size)
            pos, neg, embList = self.mdl(pos_head_ids_tensor, pos_tail_ids_tensor, pos_rel_tensor, neg_head_ids_tensor, neg_tail_ids_tensor, neg_rel_tensor, hidden)
            
            """
            if self.args.experiment == "BERT":
                output = self.mdl(input_ids_tensor, masked_idx1, masked_idx2)
            elif self.args.experiment == "LSTM":
                hidden = self.mdl.init_hidden(actual_batch_size)
                output = self.mdl(input_ids_tensor, hidden)
            """
            pos_head_embList += tensor_to_numpy(embList[0]).tolist()
            pos_tail_embList += tensor_to_numpy(embList[1]).tolist()
            pos_rel_embList += tensor_to_numpy(embList[2]).tolist()
            neg_head_embList += tensor_to_numpy(embList[3]).tolist()
            neg_tail_embList += tensor_to_numpy(embList[4]).tolist()
            neg_rel_embList += tensor_to_numpy(embList[5]).tolist()

        results = []
        for i in range(0, len(pos_head_embList)):
            pos_d = distance(pos_head_embList[i], pos_tail_embList[i], pos_rel_embList[i])
            neg_d = distance(neg_head_embList[i], neg_tail_embList[i], neg_rel_embList[i])
            if pos_d < neg_d:
                results.append(1)
            else:
                results.append(0)

        return pos_head_embList, pos_tail_embList, pos_rel_embList, float(sum(results))/float(len(results))

        
    
def model_main(args, tokenizer):
    exp = Experiment(args, tokenizer)
    print("Training...")
    final_results_str = exp.train()
    torch.save(exp.mdl.state_dict(), "trained_model.pt")
    return final_results_str
