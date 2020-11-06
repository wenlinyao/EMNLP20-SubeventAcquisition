import sys, argparse, random, glob, os, pickle, time, math, ast
sys.path.append("../utilities/")
from pytorch_transformers import *
from utilities import EventPair, extract_valid_pairs, remove_brackets
import numpy as np
from tqdm import tqdm

def read_test_pairs(file):
    input_lines = open(file, "r")
    test_all_pairs = []
    for line in input_lines:
        if not line.strip():
            continue
        words = line.split()
        if words[0] == "<filename>":
            filename = words[1]
            continue
        if words[0] == "<event1>":
            event1 = "< " + " ".join(words[1:]) + " >"
            continue
        if words[0] == "<event2>":
            event2 = "< " + " ".join(words[1:]) + " >"
            continue
        if words[0] == "<masked_sentence1>":
            masked_sentence1 = " ".join(words[1:])
            continue
        if words[0] == "<masked_sentence2>":
            masked_sentence2 = " ".join(words[1:])
            continue
        if words[0] == "<END>":
            pair = event1 + " -> " + event2
            test_all_pairs.append(pair.lower())

    input_lines.close()
    return list(set(test_all_pairs))

def read_EventEx_pairs(file):
    input_lines = open(file, "r")
    test_all_pairs = []
    for line in input_lines:
        if not line.strip():
            continue
        words = line.split()
        if words[0] == "<filename>":
            filename = words[1]
            continue
        if words[0] == "<event>":
            event = "< " + " ".join(words[1:]) + " >"
            continue
        if words[0] == "<END>":
            pair = event + " -> " + event
            test_all_pairs.append(pair.lower())

    input_lines.close()
    return list(set(test_all_pairs))


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

def corrupt_head_filter(pair, headTotal, all_pairs):
    eventpair = EventPair(pair, -1)
    while(True):
        random_idx = random.randint(0, len(headTotal)-1)
        if headTotal[random_idx] + " -> " + eventpair.event2 not in all_pairs:
            break
    return headTotal[random_idx] + " -> " + eventpair.event2

def corrupt_tail_filter(pair, tailTotal, all_pairs):
    eventpair = EventPair(pair, -1)
    while(True):
        random_idx = random.randint(0, len(tailTotal)-1)
        if eventpair.event1 + " -> " + tailTotal[random_idx] not in all_pairs:
            break
    return eventpair.event1 + " -> " + tailTotal[random_idx]

def get_ids(args, word_index, event_str):
    event_wordList = []
    for idx, w in enumerate(event_str.split()):
        if w in ["<", ">"]:
            continue
        if w[0] == "[":
            anchor_idx = idx
        event_wordList.append(w.replace("[","").replace("]",""))
    word_ids, input_mask = seq_padding(args, [word_index[x] for x in event_wordList])
    anchor_idx = min(args.max_seq_length-1, anchor_idx)
    return word_ids, anchor_idx

def make_instance(args, pair, headTotal, tailTotal, all_pairs, word_index):
    instance = {"pos_pair": pair}
    if random.random() < 0.5:
        instance["neg_pair"] = corrupt_head_filter(pair, headTotal, all_pairs)
    else:
        instance["neg_pair"] = corrupt_tail_filter(pair, tailTotal, all_pairs)
    pos_eventpair = EventPair(instance["pos_pair"], -1)
    neg_eventpair = EventPair(instance["neg_pair"], -1)
    instance["pos_head_ids"] = get_ids(args, word_index, pos_eventpair.event1)[0]
    instance["pos_tail_ids"] = get_ids(args, word_index, pos_eventpair.event2)[0]
    instance["pos_rel"] = 0
    instance["neg_head_ids"] = get_ids(args, word_index, neg_eventpair.event1)[0]
    instance["neg_tail_ids"] = get_ids(args, word_index, neg_eventpair.event2)[0]
    instance["neg_rel"] = 0

    return instance

def LSTM_prepare_data(args, all_pairs, test_all_pairs):
    trainList = []
    devList = []
    testList = []
    
    headTotal = []
    tailTotal = []
    vocab = []

    input_lines = open("test_pairs.csv", "r")
    for line in input_lines:
        event = line.split()[0]
        test_all_pairs.append("< [" + event + "] > -> < [" + event + "] >")
    input_lines.close()

    all_pairs_list = list(all_pairs)

    for pair in all_pairs_list + test_all_pairs:
        eventpair = EventPair(pair, -1)
        for w in pair.split():
            if w in ["<", ">"]:
                continue
            vocab.append(w.replace("[","").replace("]",""))
        headTotal.append(eventpair.event1)
        tailTotal.append(eventpair.event2)

    vocab = list(set(vocab))
    headTotal = list(set(headTotal))
    tailTotal = list(set(tailTotal))

    print("{} unique words".format(len(vocab)))
    print("len(all_pairs):", len(all_pairs))

    index_word = {index+2:word for index,word in enumerate(vocab)}
    word_index = {word:index+2 for index,word in enumerate(vocab)}
    index_word[0], index_word[1] = '<pad>','<unk>'
    word_index['<pad>'], word_index['<unk>'] = 0,1

    split_num = len(all_pairs_list) // 50
    

    for repeat in range(0, 100):
        for pair in all_pairs_list[split_num:]:
            instance = make_instance(args, pair, headTotal, tailTotal, all_pairs, word_index)
            trainList.append(instance)

    for repeat in range(0, 10):
        for pair in all_pairs_list[:split_num]:
            instance = make_instance(args, pair, headTotal, tailTotal, all_pairs, word_index)
            devList.append(instance)

    for pair in test_all_pairs:
        instance = make_instance(args, pair, headTotal, tailTotal, all_pairs, word_index)
        testList.append(instance)



    print("len(trainList):", len(trainList))
    print("len(devList):", len(devList))
    print("len(testList):", len(testList))


    glove = {}
    print("Read Glove embedding...")

    with open(args.w2v_file) as f:
        for l in f:
            vec = l.split(' ')
            word = vec[0].lower()
            vec = vec[1:]
            glove[word] = np.array(vec)
    vocab_size = len(vocab)
    dimensions = 300
    matrix = np.zeros((len(word_index), dimensions))
    oov = 0
    filtered_glove = {}
    for i in tqdm(range(2, len(word_index))):
        word = index_word[i].lower()
        if(word in glove):
            vec = glove[word]
            filtered_glove[word] = glove[word]
            matrix[i] = vec
        else:
            random_init = np.random.uniform(low=-0.1,high=0.1, size=(1,dimensions))
            matrix[i] = random_init
            oov +=1
    print("oov={}".format(oov))
    env = {"index_word":index_word, "word_index":word_index, "glove": matrix}

    random.shuffle(trainList)

    env["train"] = trainList
    env["dev"] = devList
    env["test"] = testList

    pickle.dump(env, open("env.pkl", "wb"))



