import sys, argparse, random, glob, os
sys.path.append("../utilities/")
import torch
import torch.nn as nn
try:
    from pytorch_transformers import *
except:
    from transformers import *

from utilities import EventPair, extract_trigger_pair2scores, extract_seed_cand_pairs_Clean, extract_seed_cand_pairs_NoClean
from utilities import extract_candidates_conj, extract_candidates_surrounding
from train import *
from model_utils import BERT_prepare_data


# bert-large-uncased
MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),  # 12-layer, 768-hidden, 12-heads, 110M parameters
          (BertModel,       BertTokenizer,      'bert-large-uncased'), # 24-layer, 1024-hidden, 16-heads, 340M parameters
          (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,      'gpt2'),
          (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024'),
          (RobertaModel,    RobertaTokenizer,   'roberta-base')]

# clean files under current directory
def clean_files():
    for file in glob.glob("*.txt"):
        os.system("rm " + file)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_num", default=3, type=int, help="Class number.")
    parser.add_argument("--max_seq_length", default=100, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--bert_hidden_size", default=768, type=int, help="BERT model hidden size.")
    parser.add_argument("--fix_embedding_layer", default=False, type=str2bool, help="Fix BERT embedding layer in training.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--cuda", default=True, type=str2bool, help="Train on GPU or not.")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU id to train models.")
    parser.add_argument("--toy", default=False, type=str2bool, help="Use toy dataset (for fast testing), True means use toy dataset")
    parser.add_argument("--prepare_data", default=False, type=str2bool, help="Prepare data (env.pkl) or not.")
    parser.add_argument("--seed", default=11, type=int, help="Random seed for initialization.")
    parser.add_argument("--mask_trigger", default=True, type=str2bool, help="Mask event trigger or not in a sentence.")
    parser.add_argument("--valid_pair_threshold", default=2, type=float, help="Score threshold for extracting good event pairs.")
    parser.add_argument("--neg_sample_rate", default=0.02, type=float, help="Negative pairs sample rate (default = 0.02).")
    parser.add_argument("--experiment", default="BERT", type=str, help="Experiment classifier.")
    parser.add_argument("--scale_weight", default=True, type=str2bool, help="Give smaller class higher weight.")
    parser.add_argument("--clean_seed", default=True, type=str2bool, help="Use cleaned seed pairs or not.")
    parser.add_argument("--genre", default="news", type=str, help="Train and predict on which genre (news or NovelBlog).")
    parser.add_argument("--eval_dataset", default="RED", type=str, help="Which dataset to evaluate BERT model (RED/HiEve).")
    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODELS[0]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    # train and predict on which genre
    args.contexts_dirList = ["../run_extract_contexts/news/"]
    

    if args.eval_dataset == "RED":
        args.test_file = "../datasets/preprocess_RED/run_preprocess_new/RED_allRelations.txt"
    elif args.eval_dataset == "HiEve":
        args.test_file = "../datasets/preprocess_hievents/run_preprocess_new/HiEve_allRelations.txt"


    if args.clean_seed == True:
        seed_pairs, candidate_pairs = extract_seed_cand_pairs_Clean("../run_model_slpa/valid_pairs.txt", "../run_extract_event_pair_nmod2/news/sorted_parent_child2num.txt")
        
    else:
        seed_pairs, candidate_pairs = extract_seed_cand_pairs_NoClean("../run_extract_event_pair_nmod2/news/sorted_parent_child2num.txt", 2.0)
        


    print("seed_pairs:", len(seed_pairs))
    candidate_pairs_conj = extract_candidates_conj("../run_extract_event_pair_conj2/news/sorted_parent_child2num.txt", 1.0)
    candidate_pairs_surrounding = extract_candidates_surrounding("../run_extract_surrounding_subevents/extract_subevent_pairs.txt", 1.0)
    print("nmod candidate pairs:", len(candidate_pairs))
    print("conj candidate pairs:", len(candidate_pairs_conj))
    print("surrounding candidate pairs:", len(candidate_pairs_surrounding))

    candidate_pairs = candidate_pairs | candidate_pairs_conj | candidate_pairs_surrounding
    print("total candidate pairs:", len(candidate_pairs))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clean_files()

    if torch.cuda.is_available() and args.cuda == True:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
        print("Using GPU device:{}".format(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
    args.device = device

    if args.prepare_data == True:
        BERT_prepare_data(args, tokenizer, seed_pairs, candidate_pairs)
    
    model_main(args, tokenizer)