import sys, argparse, random, glob, os
sys.path.append("../utilities/")
import torch
import torch.nn as nn
from pytorch_transformers import *
from utilities import EventPair, extract_valid_pairs, extract_BERT_predicted_pairs, extract_all_pairs
from train import *
from model_utils import LSTM_prepare_data, read_test_pairs, read_EventEx_pairs

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


# python ../model_Trans/LSTM_context_model_main.py --prepare_data True --epochs 20 --gpu_id 0 --rnn_size 50
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation_total", default=1, type=int, help="Relation number.")
    parser.add_argument("--rnn_size", default=300, type=int, help="RNN dimension.")
    parser.add_argument("--aggregation", default='max', type=str, help="The aggregation method for regp and bregp types (mean|last|max) (default=max)")
    parser.add_argument("--embedding_size", default=300, type=int, help="Embeddings dimension (default=300)")
    parser.add_argument("--rnn_layers", default=1, type=int, help="Number of RNN layers (default = 1)")
    parser.add_argument("--trainable", default=True, type=str2bool, help="Trainable Word Embeddings (default=False)")
    parser.add_argument("--max_seq_length", default=5, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--cuda", default=True, type=str2bool, help="Train on GPU or not.")
    parser.add_argument("--gpu_id", default=1, type=int, help="GPU id to train models.")
    parser.add_argument("--toy", default=False, type=str2bool, help="Use toy dataset (for fast testing), True means use toy dataset")
    parser.add_argument("--prepare_data", default=False, type=str2bool, help="Prepare data (env.pkl) or not.")
    parser.add_argument("--seed", default=11, type=int, help="Random seed for initialization.")
    parser.add_argument("--valid_pair_threshold", default=2, type=float, help="Score threshold for extracting good event pairs.")
    parser.add_argument("--experiment", default="LSTM", type=str, help="Experiment classifier.")
    parser.add_argument("--scale_weight", default=True, type=str2bool, help="Give smaller class higher weight.")
    parser.add_argument("--clean_seed", default=True, type=str2bool, help="Use cleaned seed pairs or not.")
    args = parser.parse_args()

    args.w2v_file = "../../tools/glove.6B/glove.6B.300d.txt"

    all_pairs_news = extract_all_pairs("../annotation/valid_pairs_v3.txt") | extract_all_pairs("../annotation/invalid_pairs_v3.txt")
    all_pairs = all_pairs_news
    
    print("all pairs:", len(all_pairs))

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
        test_all_pairs = read_test_pairs("../datasets/preprocess_EventStoryLine/run_preprocess_new/EventStoryLine_allRelations.txt") \
                            + read_test_pairs("../datasets/preprocess_hievents/run_preprocess_new/HiEve_allRelations.txt") \
                            + read_test_pairs("../datasets/preprocess_RED/run_preprocess_new/RED_allRelations.txt") \
                            + read_test_pairs("../datasets/preprocess_Timebank/run_preprocess_new/Timebank_allRelations_new.txt")
        LSTM_prepare_data(args, all_pairs, test_all_pairs)
    
    model_main(args, None)