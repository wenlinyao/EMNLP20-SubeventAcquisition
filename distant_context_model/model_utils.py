import sys, argparse, random, glob, os, pickle, time, math, ast
sys.path.append("../utilities/")
from pytorch_transformers import *
from utilities import EventPair, extract_valid_pairs, remove_brackets
import numpy as np
from tqdm import tqdm


def seq_padding(args, input_ids):
    pad_token = 0

    if len(input_ids) < args.max_seq_length:
        padding_length = args.max_seq_length - len(input_ids)
    else:
        padding_length = 0
        input_ids = input_ids[:args.max_seq_length]
    
    input_mask = [1] * len(input_ids) + [0] * padding_length
    input_ids = input_ids + [pad_token] * padding_length

    #print(input_ids)

    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length

    return input_ids, input_mask

def pair_str2instance(args, pair_str, tokenizer):
    eventpair = EventPair(pair_str, -1)
    event_pair_sentence = "[CLS] " + remove_brackets(eventpair.event1) + " [SEP] " + remove_brackets(eventpair.event2) + " [SEP]"
    
    input_ids = tokenizer.encode(event_pair_sentence)
    input_ids, input_mask = seq_padding(args, input_ids)

    event1_trigger = remove_brackets(eventpair.event1_trigger)
    event2_trigger = remove_brackets(eventpair.event2_trigger)

    event_pair_sentence_wordList = event_pair_sentence.split()

    E1_trigger_index, E2_trigger_index = event_pair_sentence_wordList.index(event1_trigger), event_pair_sentence_wordList.index(event2_trigger)

    trigger1_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event1_trigger)]
    trigger2_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event2_trigger)]

    masked_idxList = [E1_trigger_index, E2_trigger_index+len(trigger1_ids)-1]

    if masked_idxList[1] >= len(input_ids):
        print("Type2: One common-sense pair instance exceeds max_seq_length.")
        return None

    instance = {"event_pair": pair_str, "masked_sentence": event_pair_sentence, "input_ids": input_ids, "masked_idxList": masked_idxList}
    
    if " -> " in pair_str or " CONTAINS-SUBEVENT " in pair_str:
        instance["class"] = 1
    elif " <- " in pair_str or " R_CONTAINS-SUBEVENT " in pair_str:
        instance["class"] = 2
    else:
        instance["class"] = 0

    return instance

def get_class_idx(args, relation, order_flag):
    idx = None
    
    valid_relationList = ["CONTAINS-SUBEVENT", "CONTAINS", "SuperSub"]

    if relation in valid_relationList:
        idx = 1

    if idx != None:
        if order_flag == "e1->e2":
            idx += 0
        else:
            idx += 1
    else:
        idx = 0
    return idx



def BERT_prepare_data(args, tokenizer, seed_pairs, candidate_pairs):
    MASK_id = tokenizer._convert_token_to_id("[MASK]")

    filename2instanceList = {}
    input_lines = open(args.test_file, "r")
    for line in input_lines:
        if not line.strip():
            continue
        words = line.split()
        if words[0] == "<filename>":
            filename = words[-1]
            continue
        if words[0] == "<relation>":
            relation = words[1]
            continue
        if words[0] == "<event1_trigger>":
            event1_trigger = words[-1]
            continue
        if words[0] == "<event2_trigger>":
            event2_trigger = words[-1]
            continue
        if words[0] == "<order_flag>":
            order_flag = words[1]
            continue
        if words[0] == "<event1>":
            event1 = " ".join(words[1:])
            continue
        if words[0] == "<event2>":
            event2 = " ".join(words[1:])
            continue
        if words[0] == "<sentence1>":
            sentence1 = " ".join(words[1:])
            continue
        if words[0] == "<sentence2>":
            sentence2 = " ".join(words[1:])
            continue
        if words[0] == "<masked_sentence1>":
            masked_sentence1 = " ".join(words[1:])
            continue
        if words[0] == "<masked_sentence2>":
            masked_sentence2 = " ".join(words[1:])
            continue
        if words[0] == "<END>":
            if masked_sentence1 != masked_sentence2:
                continue
            instance = {}

            if order_flag == "e1->e2":
                word_pair = "< " + event1 + " > " + relation + " < " + event2 + " >"
            else:
                word_pair = "< " + event2 + " > " + "R_" + relation + " < " + event1 + " >"
            
            masked_sentence = "[CLS] " + masked_sentence1 + " [SEP]"
            input_ids = tokenizer.encode(masked_sentence)
            input_ids, input_mask = seq_padding(args, input_ids)

            masked_idxList = []

            for i, input_id in enumerate(input_ids):
                if input_id == MASK_id:
                    masked_idxList.append(i)
            if len(masked_idxList) != 2:
                print("Type1: One test instance exceeds max_seq_length.", len(masked_sentence.split()))
                print(masked_sentence + "\n")
                continue
            
            if args.mask_trigger == True:
                instance = {"event_pair": word_pair, "masked_sentence": masked_sentence, "input_ids": input_ids, "masked_idxList": masked_idxList}
            
            else:
                sentence = "[CLS] " + sentence1 + " [SEP]"
                input_ids = tokenizer.encode(sentence)
                input_ids, input_mask = seq_padding(args, input_ids)

                trigger1_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event1_trigger)]
                trigger2_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event2_trigger)]

                if order_flag == "e1->e2":
                    masked_idxList[1] = masked_idxList[1] + (len(trigger1_ids) - 1) # How many slots shift to the right
                else:
                    masked_idxList[1] = masked_idxList[1] + (len(trigger2_ids) - 1) # How many slots shift to the right
                
                if masked_idxList[1] >= len(input_ids):
                    print("Type2: One RED test instance exceeds max_seq_length.")
                    continue

                instance = {"event_pair": word_pair, "masked_sentence": masked_sentence, "input_ids": input_ids, "masked_idxList": masked_idxList}
            

            class_idx = get_class_idx(args, relation, order_flag)

            if filename not in filename2instanceList:
                filename2instanceList[filename] = []
            
            instance_found_idx = None
            for idx, previous_instance in enumerate(filename2instanceList[filename]):
                if previous_instance["masked_sentence"] == masked_sentence:
                    instance_found_idx = idx
            if instance_found_idx != None:
                if class_idx in [1, 2]: # relation and R_relation class_idx
                    filename2instanceList[filename][instance_found_idx]["event_pair"] = word_pair
                    filename2instanceList[filename][instance_found_idx]["class"] = class_idx
            else:
                instance["class"] = class_idx
                filename2instanceList[filename].append(instance)

    input_lines.close()

    official_test_filenames = ["NYT_ENG_20130426.0143", "dd0b65f632f64369c530f9bbb4b024b4.mpdf", "c06e8bbdf69f73a69cd3d5dbb4d06a21.mpdf",\
                        "NYT_ENG_20130709.0087", "NYT_ENG_20130712.0047", "NYT_ENG_20131003.0269", "PROXY_AFP_ENG_20020210_0074",\
                        "PROXY_AFP_ENG_20020408_0348", "d21dc2cb6e6435da7f9d9b0e5759e214", "soc.culture.iraq_20050211.0445"]


    trainList = []
    devList = []
    testList = []
    candidateList = []

    for contexts_dir in args.contexts_dirList:
        for input_file in glob.glob(contexts_dir + "*.txt"):
        #for input_file in glob.glob(contexts_dir + "*199407_*.txt"):
        #for input_file in glob.glob(contexts_dir + "*3_*.txt"):
            if args.genre == "news":
                print(input_file)
            input_lines = open(input_file, "r")
            for line in input_lines:
                if not line.strip():
                    continue
                words = line.split()
                if words[0] == "<word_pair>":
                    word_pair = " ".join(words[1:])
                    continue
                if words[0] == "<trigger_index>":
                    E1_trigger_index, E2_trigger_index = int(words[1]), int(words[2])
                    continue
                if words[0] == "<masked_sentence>":
                    masked_sentence = " ".join(words[1:])
                    continue
                if words[0] == "<sentence>":
                    if len(words) >= 256: # specified maximum sequence length 512
                        continue
                    instance = {"event_pair": word_pair, "masked_sentence": masked_sentence, "words": words, "trigger_index": [E1_trigger_index, E2_trigger_index]}
                    if " -> " in word_pair:
                        instance["class"] = 1
                    elif " <- " in word_pair:
                        instance["class"] = 2
                    elif " <-> " in word_pair:
                        instance["class"] = 0
                    elif " => " in word_pair: # Temporal causal pairs
                        instance["class"] = 0
                    elif " <= " in word_pair:
                        instance["class"] = 0

                    if "_neg_" in input_file:
                        if random.uniform(0, 1) <= 0.05:
                            #if "_nyt_" in input_file:
                            trainList.append(instance)
                    elif "_TC_" in input_file or "_CT_" in input_file:
                        #if "_nyt_" in input_file:
                        trainList.append(instance)
                    elif word_pair in seed_pairs:
                        #if "_nyt_" in input_file:
                        trainList.append(instance)
                    elif word_pair in candidate_pairs:
                        candidateList.append(instance)

    new_trainList = []
    pair2num = {}
    random.shuffle(trainList)
    for instance in trainList:
        if instance["class"] == 0:
            new_trainList.append(instance)
            continue
        event_pair = instance["event_pair"]
        if event_pair not in pair2num:
            pair2num[event_pair] = 1
        else:
            pair2num[event_pair] += 1
        if pair2num[event_pair] <= 10:
            new_trainList.append(instance)
    observed_seed_pairs_count = len(pair2num.keys())
    trainList = new_trainList

    new_candidateList = []
    pair2num = {}
    random.shuffle(candidateList)
    for instance in candidateList:
        event_pair = instance["event_pair"]
        if event_pair not in pair2num:
            pair2num[event_pair] = 1
        else:
            pair2num[event_pair] += 1
        if pair2num[event_pair] <= 20:
            new_candidateList.append(instance)
    candidateList = new_candidateList

    new_trainList = []
    new_candidateList = []
    for flag, process_list in enumerate([trainList, candidateList]):
        for instance in process_list:
            if args.mask_trigger == True:
                masked_sentence = instance["masked_sentence"]
                masked_sentence = "[CLS] " + masked_sentence + " [SEP]"
                input_ids = tokenizer.encode(masked_sentence)
                input_ids, input_mask = seq_padding(args, input_ids)

                masked_idxList = []

                for i, input_id in enumerate(input_ids):
                    if input_id == MASK_id:
                        masked_idxList.append(i)
                if len(masked_idxList) == 2:
                    instance["masked_sentence"] = masked_sentence 
                    instance["input_ids"] = input_ids
                    instance["masked_idxList"] = masked_idxList

                    if len(instance["input_ids"]) < 256:
                        if flag == 0:
                            new_trainList.append(instance)
                        else:
                            new_candidateList.append(instance)
                    
            else:
                masked_sentence = instance["masked_sentence"]
                masked_sentence = "[CLS] " + masked_sentence + " [SEP]"
                words = instance["words"]
                sentence = "[CLS] " + " ".join(words[1:]) + " [SEP]"
                input_ids = tokenizer.encode(sentence) # Use original sentence.
                input_ids, input_mask = seq_padding(args, input_ids)
                E1_trigger_index, E2_trigger_index = instance["trigger_index"]
                event1_trigger = words[E1_trigger_index]
                event2_trigger = words[E2_trigger_index]

                trigger1_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event1_trigger)]
                trigger2_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event2_trigger)]

                masked_idxList = [E1_trigger_index, E2_trigger_index+len(trigger1_ids)-1]

                if masked_idxList[1] >= len(input_ids):
                    print("Type2: One Gigaword instance exceeds max_seq_length.")
                    continue

                instance["masked_sentence"] = masked_sentence 
                instance["input_ids"] = input_ids
                instance["masked_idxList"] = masked_idxList

                if len(instance["input_ids"]) < 256:
                    if flag == 0:
                        new_trainList.append(instance)
                    else:
                        new_candidateList.append(instance)

    trainList = new_trainList
    candidateList = new_candidateList
    
    for filename in filename2instanceList:
        testList += filename2instanceList[filename]
        
    random.shuffle(trainList)

    split_num = len(trainList) // 10

    devList = trainList[:split_num]
    trainList = trainList[split_num:]

    print("{} train sentences".format(len(trainList)))
    print("{} candidate sentences".format(len(candidateList)))
    print("len(seed_pairs):", len(seed_pairs))
    print("len(candidate_pairs):", len(candidate_pairs))
    print("Observed seed pairs:", observed_seed_pairs_count)

    env = {}
    env["train"] = trainList
    env["dev"] = devList
    env["test"] = testList
    env["candidate"] = candidateList

    pickle.dump(env, open("env.pkl", "wb"))

pattern_set = set(["prep_during", "prep_in", "prep_amid", "prep_throughout", "prep_including", "prep_within",
                "R-prep_during", "R-prep_in", "R-prep_amid", "R-prep_throughout", "R-prep_including", "R-prep_within"])




