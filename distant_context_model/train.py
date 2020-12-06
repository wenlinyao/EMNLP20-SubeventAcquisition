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


class Experiment:
    def __init__(self, args, tokenizer):
        self.args = args
        
        self.env = pickle.load(open("env.pkl", "rb"))

        self.train_set = self.env['train']
        self.dev_set = self.env['dev']
        self.test_set = self.env['test']
        self.cand_set = self.env['candidate']

        if(self.args.toy == True):
            print("Using toy mode...")
            random.shuffle(self.train_set)
            self.train_set = self.train_set[:2000]
            self.dev_set = self.dev_set[:500]
            random.shuffle(self.test_set)
            self.test_set = self.test_set[:500]

        classes_freq = [1 for i in range(0, self.args.class_num)]
        for instance in self.train_set:
            classes_freq[instance["class"]] += 1
        classes_freq_sum = sum(classes_freq)

        #classes_weight = [math.log(float(classes_freq_sum)/float(freq)) for freq in classes_freq]
        print(self.args.scale_weight)
        if self.args.scale_weight == True:
            classes_weight = [float(classes_freq_sum)/float(freq) for freq in classes_freq]
        else:
            classes_weight = [1.0 for freq in classes_freq]

        self.classes_weight = torch.from_numpy(np.array(classes_weight, dtype='float32'))
        print("classes_freq:", classes_freq)
        print("classes_weight:", classes_weight)

        if args.experiment == "BERT":
            self.mdl = BERT_NN(args)
        elif args.experiment == "LSTM":
            self.mdl = RNN(args, len(self.env["word_index"]), pretrained=self.env["glove"])

        if self.args.cuda == True:
            self.mdl.to(self.args.device)
            self.classes_weight = self.classes_weight.to(self.args.device)

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

        input_ids_tensor = torch.tensor([instance['input_ids'] for instance in batch])
        targets = torch.LongTensor(np.array([instance['class'] for instance in batch], dtype=np.int32).tolist())
        if self.args.experiment == "BERT":
            masked_idx1 = torch.LongTensor(np.array([instance['masked_idxList'][0] for instance in batch], dtype=np.int32).tolist())
            masked_idx2 = torch.LongTensor(np.array([instance['masked_idxList'][1] for instance in batch], dtype=np.int32).tolist())
        else:
            masked_idx1, masked_idx2 = None, None
        if self.args.cuda == True:
            input_ids_tensor = input_ids_tensor.to(self.args.device)
            targets = targets.to(self.args.device)
            if self.args.experiment == "BERT":
                masked_idx1 = masked_idx1.to(self.args.device)
                masked_idx2 = masked_idx2.to(self.args.device)

        actual_batch_size = input_ids_tensor.size(0)
        input_ids_tensor = Variable(input_ids_tensor, requires_grad=False)
        targets = Variable(targets, requires_grad=False)

        return input_ids_tensor, masked_idx1, masked_idx2, targets, actual_batch_size

    def train_batch(self, i):
        self.mdl.train()
        input_ids_tensor, masked_idx1, masked_idx2, targets, actual_batch_size = self.make_batch(self.train_set, i)

        #self.mdl.zero_grad()
        self.optimizer.zero_grad()

        if self.args.experiment == "BERT":
            output = self.mdl(input_ids_tensor, masked_idx1, masked_idx2)
        elif self.args.experiment == "LSTM":

            hidden = self.mdl.init_hidden(actual_batch_size)
            #print(input_ids_tensor.size())
            output = self.mdl(input_ids_tensor, hidden)
        
        #print "output:", output
        #print "targets:", targets
        loss = self.criterion(output, targets)
        
        loss.backward()

        nn.utils.clip_grad_norm_(parameters = self.mdl.parameters(), max_norm = self.args.max_grad_norm)
        self.optimizer.step()

        #return loss.data[0]
        return loss.item()

    def train(self):
        """
        This is the main train function
        """
        self.criterion = nn.CrossEntropyLoss(weight = self.classes_weight)
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

            if len(self.dev_set) != 0:
                print("Evaluate on dev set...")
                avg_P, avg_R, avg_F, results_str = self.test(epoch, "dev")
                print(results_str)
            
            if len(self.test_set) != 0:
                print("Evaluate on test set...")
                avg_P, avg_R, avg_F, results_str = self.test(epoch, "test")
                print(results_str)

        if len(self.cand_set) != 0:
            self.test(epoch, "candidate")
    
    def test(self, epoch, data_flag):
        if data_flag == "dev":
            dataset = self.dev_set
            output_file = None

        elif data_flag == "test":
            dataset = self.test_set
            output_file = open("test_TruePred_" + str(epoch) + ".txt", "w")

        elif data_flag == "candidate":
            dataset = self.cand_set
            output_file = open("candidate_TruePred_" + str(epoch) + ".txt", "w")

        all_probs, all_preds, acc, avg_P, avg_R, avg_F, results_str = self.evaluate(dataset)

        if output_file == None:
            return avg_P, avg_R, avg_F, results_str

        for i, instance in enumerate(dataset):
            if "word_path" in instance:
                output_file.write(instance["event_pair"] + "\t" + " ".join(instance["word_path"]) + "\t" + str(instance["class"]) + "\t" + str(all_preds[i]) + "\t" + str(all_probs[i]) + "\n")
            else:
                output_file.write(instance["event_pair"] + "\t" + str(instance["class"]) + "\t" + str(all_preds[i]) + "\t" + str(all_probs[i]) + "\n")
        output_file.close()
        return avg_P, avg_R, avg_F, results_str

    def evaluate(self, x):
        self.mdl.eval()

        if len(x) % self.args.batch_size == 0:
            num_batches = int(len(x) / self.args.batch_size)
        else:
            num_batches = int(len(x) / self.args.batch_size) + 1

        all_probs = []
        all_preds = []
        all_targets = []
        
        for instance in x:
            all_targets.append(instance["class"])

        for i in range(num_batches):
            input_ids_tensor, masked_idx1, masked_idx2, targets, actual_batch_size = self.make_batch(x, i)
            if self.args.experiment == "BERT":
                output = self.mdl(input_ids_tensor, masked_idx1, masked_idx2)
            elif self.args.experiment == "LSTM":
                hidden = self.mdl.init_hidden(actual_batch_size)
                output = self.mdl(input_ids_tensor, hidden)
            all_probs += tensor_to_numpy(output).tolist()

        for probs in all_probs:
            all_preds.append(probs.index(max(probs)))
        
        # print("len(all_targets):", len(all_targets), "len(all_preds):", len(all_preds))
        confusion_matrix = {}
        matches = 0
        for i in range(len(all_targets)):
            if all_targets[i] == all_preds[i]:
                matches += 1
            string = str(all_targets[i]) + " --> " + str(all_preds[i])
            if string in confusion_matrix:
                confusion_matrix[string] += 1
            else:
                confusion_matrix[string] = 1
        acc = float(matches) / float(len(all_targets))
        print("accuracy:", acc)
        print("confusion_matrix[target --> pred]:", confusion_matrix)
        labelList = [label for label in range(0, self.args.class_num)]
        
        results = precision_recall_fscore_support(all_targets, all_preds, average=None, labels=labelList)
        results_str = []
        results_str.append("\t".join([str(l) for l in labelList]))
        results_str.append("\t".join(["%0.4f" % p for p in results[0]]))
        results_str.append("\t".join(["%0.4f" % r for r in results[1]]))
        results_str.append("\t".join(["%0.4f" % f for f in results[2]]))
        results_str = "\n".join(results_str)
        avg = precision_recall_fscore_support(all_targets, all_preds, average='macro')
        avg_P, avg_R, avg_F = avg[0], avg[1], avg[2]
        #print(results_str)

        return all_probs, all_preds, acc, avg_P, avg_R, avg_F, results_str
    
def model_main(args, tokenizer):
    exp = Experiment(args, tokenizer)
    print("Training...")
    final_results_str = exp.train()
    torch.save(exp.mdl.state_dict(), "trained_model.pt")
    return final_results_str
