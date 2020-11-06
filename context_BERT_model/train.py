import sys, argparse, random, tqdm, pickle, time, glob
sys.path.append("../utilities/")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
try:
    from pytorch_transformers import *
except:
    from transformers import *
from models import *
from utilities import EventPair, remove_brackets
from evaluate_sep_withinSent import evaluate_sep

def tensor_to_numpy(x):
    ''' Need to cast before calling numpy()
    '''
    #return (Variable(x).data).cpu().numpy()
    return x.data.type(torch.DoubleTensor).numpy()


class Experiment:
    def __init__(self, args, tokenizer):
        random.seed(111)
        torch.manual_seed(111)
        np.random.seed(111)

        # random.seed(11)
        # torch.manual_seed(11)
        # np.random.seed(11)

        self.args = args
        
        self.env = pickle.load(open("env.pkl", "rb"))
        self.org_train_set = self.env['train']
        self.train_set = self.env['train']
        self.dev_set = self.env['dev']
        self.test_set = self.env['test']
        # self.candidate_set = self.env['candidate']
        if self.args.crossDomain == True:
            if self.args.cross_id == "0":
                self.filename2instanceList = pickle.load(open("HiEve_filename2instanceList.p", "rb"))
            elif self.args.cross_id == "1":
                self.filename2instanceList = pickle.load(open("RED_filename2instanceList.p", "rb"))
        else:
            self.filename2instanceList = pickle.load(open("filename2instanceList.p", "rb"))

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
        classes_weight = [float(classes_freq_sum)/float(freq) for freq in classes_freq]
        #classes_weight[0] /= 2.0
        #classes_weight = [1.0 for freq in classes_freq]

        self.classes_weight = torch.from_numpy(np.array(classes_weight, dtype='float32'))
        print("classes_freq:", classes_freq)
        print("classes_weight:", classes_weight)

        #print(self.train_set[:10])

        """
        # Use multi-GPU
        if torch.cuda.device_count() > 1:
            self.mdl = nn.DataParallel(BasicNN(args))
            self.mdl.to(self.args.device)
            self.classes_weight = self.classes_weight.to(self.args.device)
        elif torch.cuda.is_available():
            self.mdl = BasicNN(args)
            self.mdl.to(self.args.device)
            self.classes_weight = self.classes_weight.to(self.args.device)
        """
        self.mdl = BasicNN(args)

        if self.args.cuda == True:
            self.mdl.to(self.args.device)
            self.classes_weight = self.classes_weight.to(self.args.device)
        

    def select_optimizer(self):
        parameters = filter(lambda p: p.requires_grad, self.mdl.parameters())
        self.optimizer =  optim.Adam(parameters, lr=self.args.learning_rate)

    def make_new_train_set(self):
        new_train_set = []
        for instance in self.org_train_set:
            if instance["class"] == 0:
                if random.uniform(0, 1) <= 0.2:
                    new_train_set.append(instance)
            else:
                new_train_set.append(instance)
        self.train_set = new_train_set


    
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
        knowledge_vector_tensor = torch.tensor([instance['knowledge_vector'] for instance in batch])
        children_vector_tensor = torch.tensor([instance['children_vector'] for instance in batch], dtype=torch.float)
        targets = torch.LongTensor(np.array([instance['class'] for instance in batch], dtype=np.int32).tolist())
        masked_idx1 = torch.LongTensor(np.array([instance['masked_idxList'][0] for instance in batch], dtype=np.int32).tolist())
        masked_idx2 = torch.LongTensor(np.array([instance['masked_idxList'][1] for instance in batch], dtype=np.int32).tolist())

        if self.args.cuda == True:
            input_ids_tensor = input_ids_tensor.to(self.args.device)
            knowledge_vector_tensor = knowledge_vector_tensor.to(self.args.device)
            children_vector_tensor = children_vector_tensor.to(self.args.device)
            targets = targets.to(self.args.device)
            masked_idx1 = masked_idx1.to(self.args.device)
            masked_idx2 = masked_idx2.to(self.args.device)

        actual_batch_size = input_ids_tensor.size(0)
        input_ids_tensor = Variable(input_ids_tensor, requires_grad=False)
        knowledge_vector_tensor = Variable(knowledge_vector_tensor, requires_grad=False)
        children_vector_tensor = Variable(children_vector_tensor, requires_grad=False)
        targets = Variable(targets, requires_grad=False)

        return input_ids_tensor, knowledge_vector_tensor, children_vector_tensor, masked_idx1, masked_idx2, targets, actual_batch_size

    def train_batch(self, i):
        self.mdl.train()
        input_ids_tensor, knowledge_vector_tensor, children_vector_tensor, masked_idx1, masked_idx2, targets, actual_batch_size = self.make_batch(self.train_set, i)

        #self.mdl.zero_grad()
        self.optimizer.zero_grad()

        output = self.mdl(input_ids_tensor, knowledge_vector_tensor, children_vector_tensor, masked_idx1, masked_idx2)
        
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
        #self.criterion = nn.NLLLoss(weight = self.classes_weight)
        print(self.args)
        total_loss = 0

        
        print("len(self.train_set)", len(self.train_set))
        
        self.select_optimizer()

        best_acc = 0
        best_F = 0
        final_results_strList = []
        for epoch in range(1, self.args.epochs+1):
            self.mdl.train()
            print("epoch: ", epoch)
            t0 = time.clock()
            #self.make_new_train_set()
            random.shuffle(self.train_set)

            if len(self.train_set) % self.args.batch_size == 0:
                num_batches = int(len(self.train_set) / self.args.batch_size)
            else:
                num_batches = int(len(self.train_set) / self.args.batch_size) + 1
            print("num_batches:", num_batches)

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
                self.test(epoch, "dev")

            print("Evaluate on test set...")
            avg_P, avg_R, avg_F, results_str = self.test(epoch, "test")
            #if avg_F > best_F:
            #    best_F = avg_F
            final_results_strList.append(results_str)
        return final_results_strList # return best model performance on test data
    
    def test(self, epoch, data_flag):
        if data_flag == "dev":
            dataset = self.dev_set
            output_file = None

        elif data_flag == "test":
            dataset = self.test_set
            if self.args.knowledge_vector == False and self.args.children_vector == False:
                output_file_name = self.args.cross_id + "_S1_true_and_pred_value_" + str(epoch) + ".txt"
            elif self.args.knowledge_vector == True and self.args.children_vector == False:
                output_file_name = self.args.cross_id + "_S2_true_and_pred_value_" + str(epoch) + ".txt"
            elif self.args.knowledge_vector == False and self.args.children_vector == True:
                output_file_name = self.args.cross_id + "_S3_true_and_pred_value_" + str(epoch) + ".txt"
            elif self.args.knowledge_vector == True and self.args.children_vector == True:
                output_file_name = self.args.cross_id + "_S4_true_and_pred_value_" + str(epoch) + ".txt"
            output_file = open(output_file_name, "w")

        all_probs, all_preds, acc, avg_P, avg_R, avg_F, results_str = self.evaluate(dataset)

        if output_file == None:
            return

        for i, instance in enumerate(dataset):
            output_file.write(instance["event_pair"] + "\t" + str(instance["class"]) + "\t" + str(all_preds[i]) + "\t" + str(all_probs[i]) + "\n")
        output_file.close()

        labelList = [str(label) for label in range(0, self.args.class_num)]

        new_results_str = evaluate_sep(labelList, self.filename2instanceList, output_file_name)

        return avg_P, avg_R, avg_F, new_results_str

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
            event_pair_tensor, knowledge_vector_tensor, children_vector_tensor, masked_idx1, masked_idx2, targets, actual_batch_size = self.make_batch(x, i)
            output = self.mdl(event_pair_tensor, knowledge_vector_tensor, children_vector_tensor, masked_idx1, masked_idx2)
            output = nn.functional.softmax(output)
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
        micro_avg = precision_recall_fscore_support(all_targets, all_preds, average='micro', labels=labelList[1:])
        results_str = []
        results_str.append("\t".join([str(l) for l in labelList]) + "\t" + "1_2_micro")
        results_str.append("\t".join(["%0.4f" % p for p in results[0]]) + "\t" + "%0.4f"%micro_avg[0])
        results_str.append("\t".join(["%0.4f" % r for r in results[1]]) + "\t" + "%0.4f"%micro_avg[1])
        results_str.append("\t".join(["%0.4f" % f for f in results[2]]) + "\t" + "%0.4f"%micro_avg[2])
        results_str = "\n".join(results_str)
        avg = precision_recall_fscore_support(all_targets, all_preds, average='macro')
        avg_P, avg_R, avg_F = avg[0], avg[1], avg[2]
        print(results_str)

        return all_probs, all_preds, acc, avg_P, avg_R, avg_F, results_str
    
def model_main(args, tokenizer):
    exp = Experiment(args, tokenizer)
    print("Training...")
    final_results_str = exp.train()
    return final_results_str
