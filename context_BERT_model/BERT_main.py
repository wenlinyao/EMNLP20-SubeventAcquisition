import sys, argparse, random, glob, os, pickle
sys.path.append("../utilities/")
import torch
import torch.nn as nn
try:
    from pytorch_transformers import *
except:
    from transformers import *
from utilities import EventPair, extract_trigger_pair2score, extract_trigger_pair2score_new, CT_trigger_pair2score, extract_all_pairs, extract_CT_pairs
from utilities import extract_valid_pairs, extract_BERT_predicted_pairs
from train import *
from model_utils import prepare_data, get_filename2instanceList_new, find_closest_trigger, extract_parentTrigger2children_vec, extract_childTrigger2parents_vec

# python ../context_BERT_model/BERT_main.py --preprocess False --eval_dataset HiEve --eval_relation subevent --gpu_id 3 --batch_size 16 --epochs 6 --sentence_setting across --slpa_clean False

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
    parser.add_argument("--max_seq_length", default=200, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--bert_hidden_size", default=768, type=int, help="BERT model hidden size.")
    parser.add_argument("--fix_embedding_layer", default=False, type=str2bool, help="Fix BERT embedding layer in training.")
    parser.add_argument("--fix_BERT", default=False, type=str2bool, help="Fix all BERT layers in training.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--epochs", default=6, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--cuda", default=True, type=str2bool, help="Train on GPU or not.")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU id to train models.")
    parser.add_argument("--toy", default=False, type=str2bool, help="Use toy dataset (for fast testing), True means use toy dataset")
    parser.add_argument("--preprocess_data", default=False, type=str2bool, help="Preprocess data (filename2instanceList) or not.")
    parser.add_argument("--seed", default=11, type=int, help="Random seed for initialization.")
    #parser.add_argument("--experiment", default="RED_cross_val", type=str, help="Choose from distant_supervision, common_sense_pairs, RED_cross_val, RED_official_split")
    parser.add_argument("--mask_trigger", default=False, type=str2bool, help="Mask event trigger or not in a sentence.")
    parser.add_argument("--knowledge_vector", default=False, type=str2bool, help="Use knowledge vector or not.")
    parser.add_argument("--children_vector", default=False, type=str2bool, help="Use children vector or not.")
    parser.add_argument("--valid_pair_threshold", default=2, type=float, help="Score threshold for extracting good event pairs.")
    parser.add_argument("--eval_relation", default="subevent", type=str, help="Evaluate on subevent or causal relation (subevent/causal/temporal).")
    parser.add_argument("--slpa_clean", default=True, type=str2bool, help="Use slpa cleaned pairs or all pairs.")
    parser.add_argument("--sentence_setting", default="within", type=str, help="Consider pairs within sentence or across sentences (within/across).")
    parser.add_argument("--eval_dataset", default="RED", type=str, help="Which dataset to evaluate BERT model (RED/HiEve/ESL/Timebank).")
    
    args = parser.parse_args()

    args.crossDomain = False

    assert (not (args.knowledge_vector == True and args.children_vector == True))

    model_class, tokenizer_class, pretrained_weights = MODELS[0]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    args.contexts_dir = "../run_extract_contexts/news/"

    if args.eval_dataset == "RED":
        args.test_file = "../datasets/preprocess_RED/run_preprocess_new/RED_allRelations.txt"
    elif args.eval_dataset == "HiEve":
        args.test_file = "../datasets/preprocess_hievents/run_preprocess_new/HiEve_allRelations.txt"
    elif args.eval_dataset == "ESL":
        args.test_file = "../datasets/preprocess_EventStoryLine/run_preprocess_new/EventStoryLine_allRelations.txt"
    elif args.eval_dataset == "Timebank":
        args.test_file = "../datasets/preprocess_Timebank/run_preprocess_new/Timebank_allRelations_org_filename.txt"
    
    #args.knowledge_vec_size = (2*1) * 3  # multiRe
    args.knowledge_vec_size = (2*1) * 1 # subevent relation
    #args.children_vec_size = 300
    args.children_vec_size = 50

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clean_files()

    config_output = open("config.txt", "w")
    config_output.write(str(args))
    config_output.close()

    
    if torch.cuda.is_available() and args.cuda == True:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
        print("Using GPU device:{}".format(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
    args.device = device
    """
    # Use multi-GPU
    # CUDA_VISIBLE_DEVICES=2,3 python cross_validate.py 
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """

    
    valid_pairs = extract_all_pairs("../subevent_pairs/all_subevent_pairs.txt")
    trigger_pair2score = extract_trigger_pair2score_new(valid_pairs)


    print("len(valid_pairs):", len(valid_pairs))


    parentTrigger2children_vec = extract_parentTrigger2children_vec(valid_pairs)
    print("len(parentTrigger2children_vec):", len(parentTrigger2children_vec))

    childTrigger2parents_vec = extract_childTrigger2parents_vec(valid_pairs)
    print("len(childTrigger2parents_vec):", len(childTrigger2parents_vec))

    input_lines = open(args.test_file, "r")
    fileList = []
    for line in input_lines:
        if not line.strip():
            continue
        words = line.split()
        if words[0] == "<filename>":
            filename = words[-1]
            fileList.append(filename)
            continue
    input_lines.close()

    fileList = list(set(fileList))
    fileList = sorted(fileList)
    random.shuffle(fileList)
    random.shuffle(fileList)
    print("fileList:", fileList[:10])

    files_foldList = [[], [], [], [], []]
    for i in range(len(fileList)):
        fold_id = i % 5
        files_foldList[fold_id].append(fileList[i])

    output = open("results_strList.txt", "w", 1) # write immediately
    final_output = open("final_results_str.txt", "w", 1)

    if args.preprocess_data == True:
        filename2instanceList = get_filename2instanceList_new(args, trigger_pair2score, parentTrigger2children_vec, childTrigger2parents_vec, tokenizer)
        
        pickle.dump(filename2instanceList, open("filename2instanceList.p", "wb"))
    else:
        filename2instanceList = pickle.load(open("filename2instanceList.p", "rb"))

    for cross_id in range(0, 5):
        test_files = set(files_foldList[cross_id])
        args.cross_id = str(cross_id)

        # test_files = set() # use 20% to train and 80% to test
        # for i in range(0, 5):
        #     if i != cross_id:
        #         test_files = test_files | set(files_foldList[i])

        prepare_data(filename2instanceList, test_files)

        
        print("################## knowledge_vector = False, children_vector = False ###################")
        args.knowledge_vector, args.children_vector = False, False
        results_strList1 = model_main(args, tokenizer)
        
        
        print("################## knowledge_vector = True, children_vector = False ###################")
        args.knowledge_vector, args.children_vector = True, False
        results_strList2 = model_main(args, tokenizer)
        
        
        print("################## knowledge_vector = False, children_vector = True ###################")
        args.knowledge_vector, args.children_vector = False, True
        results_strList3 = model_main(args, tokenizer)
        
        
        output.write("######### knowledge_vector = False, children_vector = False ##########\n")
        final_output.write("######### knowledge_vector = False, children_vector = False ##########\n")
        final_output.write(results_strList1[-1]+"\n\n")
        for i in range(0, len(results_strList1)):
            output.write(results_strList1[i]+"\n\n")
        
        
        output.write("######### knowledge_vector = True, children_vector = False ##########\n")
        final_output.write("######### knowledge_vector = True, children_vector = False ##########\n")
        final_output.write(results_strList2[-1]+"\n\n")
        for i in range(0, len(results_strList2)):
            output.write(results_strList2[i]+"\n\n")
        
        
        output.write("######### knowledge_vector = False, children_vector = True ##########\n")
        final_output.write("######### knowledge_vector = False, children_vector = True ##########\n")
        final_output.write(results_strList3[-1]+"\n\n")
        for i in range(0, len(results_strList3)):
            output.write(results_strList3[i]+"\n\n")
        
        

    output.close()
    final_output.close()