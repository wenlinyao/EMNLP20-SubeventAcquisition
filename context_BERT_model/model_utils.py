import sys, argparse, random, glob, os, pickle, time, math, gensim, ast
sys.path.append("../utilities/")
import numpy as np
from utilities import EventPair, remove_brackets, B_light_verbs
from nltk.stem.snowball import SnowballStemmer
try:
    from pytorch_transformers import *
except:
    from transformers import *

stemmer = SnowballStemmer("english")
model = gensim.models.KeyedVectors.load_word2vec_format("../../tools/GoogleNews-vectors-negative300.bin", binary = True)

invalid_words = set(["person", "location", "be", "time", "act", "event", "activities",
                         "of", "for", "to", "up", "on", "with", "not", "at", "from", "into", "over", "by",
                         "about", "off", "before"])

# read event embeddings from a file
def read_event_vec(file):
    input_lines = open(file, "r")
    event2vec = {}
    for line in input_lines:
        fields = line.split("\t")
        event = fields[0]
        if event not in event2vec:
            event2vec[event] = np.array(ast.literal_eval(fields[1])) * 1000.0
        
    input_lines.close()
    return event2vec

# Use word2vec to find the closest trigger word
def find_closest_trigger(E_trigger, parentTrigger2children_vec):
    if E_trigger in parentTrigger2children_vec:
        return E_trigger

    E_trigger_w = E_trigger.replace("[", "").replace("]", "")
    max_sim, best_trigger = 0, None
    for parentTrigger in parentTrigger2children_vec:
        parentTrigger_w = parentTrigger.replace("[", "").replace("]", "")
        if E_trigger_w in model and parentTrigger_w in model:
            sim = model.similarity(E_trigger_w, parentTrigger_w)
            if sim >= 0.5:
                if sim > max_sim:
                    max_sim = sim
                    best_trigger = parentTrigger
    return best_trigger

# Use word stem to find the closest trigger word
def find_closest_trigger_stem(E_trigger, parentTrigger2children_vec):
    if E_trigger in parentTrigger2children_vec:
        return E_trigger

    E_trigger_w = stemmer.stem(E_trigger.replace("[", "").replace("]", ""))
    best_trigger = None
    for parentTrigger in parentTrigger2children_vec:
        parentTrigger_w = stemmer.stem(parentTrigger.replace("[", "").replace("]", ""))
        if E_trigger_w == parentTrigger_w:
            best_trigger = parentTrigger
            break
        
    return best_trigger

pair2sim = {}

def find_closest_trigger_pair(event1_trigger, event2_trigger, trigger_pair2score):
    trigger_pair = event1_trigger + " " + event2_trigger
    R_trigger_pair = event2_trigger + " " + event1_trigger

    if trigger_pair in trigger_pair2score:
        return trigger_pair, "trigger_pair"
    elif R_trigger_pair in trigger_pair2score:
        return R_trigger_pair, "R_trigger_pair"
    #else:
    #    return None, None

    
    event1_trigger_w = event1_trigger.replace("[", "").replace("]", "")
    event2_trigger_w = event2_trigger.replace("[", "").replace("]", "")

    max_sim, best_trigger_pair, trigger_pair_flag = 0, None, None
    for trigger_pair in trigger_pair2score:
        E1_trigger_w = trigger_pair.split()[0].replace("[", "").replace("]", "")
        E2_trigger_w = trigger_pair.split()[1].replace("[", "").replace("]", "")

        if event1_trigger_w in model and E1_trigger_w in model and event2_trigger_w in model and E2_trigger_w in model:
            if event1_trigger_w + " " + E1_trigger_w in pair2sim:
                sim1 = pair2sim[event1_trigger_w + " " + E1_trigger_w]
            else:
                sim1 = model.similarity(event1_trigger_w, E1_trigger_w)
                pair2sim[event1_trigger_w + " " + E1_trigger_w] = sim1

            if event2_trigger_w + " " + E2_trigger_w in pair2sim:
                sim2 = pair2sim[event2_trigger_w + " " + E2_trigger_w]
            else:
                sim2 = model.similarity(event2_trigger_w, E2_trigger_w)
                pair2sim[event2_trigger_w + " " + E2_trigger_w] = sim2

            sim = min(sim1, sim2)

            if event2_trigger_w + " " + E1_trigger_w in pair2sim:
                sim3 = pair2sim[event2_trigger_w + " " + E1_trigger_w]
            else:
                sim3 = model.similarity(event2_trigger_w, E1_trigger_w)
                pair2sim[event2_trigger_w + " " + E1_trigger_w] = sim3

            if event1_trigger_w + " " + E2_trigger_w in pair2sim:
                sim4 = pair2sim[event1_trigger_w + " " + E2_trigger_w]
            else:
                sim4 = model.similarity(event1_trigger_w, E2_trigger_w)
                pair2sim[event1_trigger_w + " " + E2_trigger_w] = sim4

            R_sim = min(sim3, sim4)
            
            if sim >= R_sim and sim > 0.5:
                if sim > max_sim:
                    max_sim = sim
                    best_trigger_pair = trigger_pair
                    trigger_pair_flag = "trigger_pair"
            elif R_sim > sim and R_sim > 0.5:
                if R_sim > max_sim:
                    max_sim = R_sim
                    best_trigger_pair = trigger_pair
                    trigger_pair_flag = "R_trigger_pair"
    return best_trigger_pair, trigger_pair_flag
    


def extract_parentTrigger2children_vec(valid_pairs):
    
    parentTrigger2children_words = {}
    for pair in valid_pairs:
        eventpair = EventPair(pair, -1)
        if eventpair.event1_trigger not in parentTrigger2children_words:
            parentTrigger2children_words[eventpair.event1_trigger] = []
        for w in eventpair.event2.split():
            if w in ["<", ">"] or w in invalid_words or w in B_light_verbs:
                continue

            parentTrigger2children_words[eventpair.event1_trigger].append(w.replace("[","").replace("]",""))

    parentTrigger2children_vec = {}
    for parentTrigger in parentTrigger2children_words:
        vec = np.zeros(300)
        count = 0
        for w in parentTrigger2children_words[parentTrigger]:
            if w in model:
                vec += model[w]
                count += 1
        if count != 0:
            vec = vec / float(count)
            parentTrigger2children_vec[parentTrigger] = vec

    return parentTrigger2children_vec

def extract_childTrigger2parents_vec(valid_pairs):
    
    childTrigger2parents_words = {}
    for pair in valid_pairs:
        eventpair = EventPair(pair, -1)
        if eventpair.event2_trigger not in childTrigger2parents_words:
            childTrigger2parents_words[eventpair.event2_trigger] = []
        for w in eventpair.event1.split():
            if w in ["<", ">"] or w in invalid_words or w in B_light_verbs:
                continue

            childTrigger2parents_words[eventpair.event2_trigger].append(w.replace("[","").replace("]",""))

    childTrigger2parents_vec = {}
    for childTrigger in childTrigger2parents_words:
        vec = np.zeros(300)
        count = 0
        for w in childTrigger2parents_words[childTrigger]:
            if w in model:
                vec += model[w]
                count += 1
        if count != 0:
            vec = vec / float(count)
            childTrigger2parents_vec[childTrigger] = vec

    return childTrigger2parents_vec


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
    
    if " -> " in pair_str or " CONTAINS-SUBEVENT " in pair_str or " CONTAINS " in pair_str or " SuperSub " in pair_str:
        instance["class"] = 1
    elif " <- " in pair_str or " R_CONTAINS-SUBEVENT " in pair_str or " R_CONTAINS " in pair_str or " R_SuperSub " in pair_str:
        instance["class"] = 2
    else:
        instance["class"] = 0

    return instance


def score2feature(score):
    """
    # I
    if score >= 2:
        return 1.0
    elif 1 <= score and score < 2:
        return 0.5
    else: 
        return 0.1
    """
    """
    if score >= 2:
        return 0.3
    elif 1 <= score and score < 2:
        return 0.2
    else: 
        return 0.1
    """
    """
    # II
    if score > 0.4:
        return 1.0
    else:
        return random.uniform(-0.01,0.01)
    """
    return score




def get_class_idx(args, relation, order_flag):
    idx = None
    if args.eval_relation == "subevent":
        valid_relationList = ["CONTAINS-SUBEVENT", "CONTAINS", "SuperSub", "INCLUDES"]
    elif args.eval_relation == "causal":
        valid_relationList = ["PRECONDITION", "CAUSES", "RISING_ACTION"]
    elif args.eval_relation == "temporal":
        valid_relationList = ["BEFORE", "IBEFORE"]
    else:
        print("args.eval_relation error.")
        return

    if relation in valid_relationList:
        idx = 1
    
    if idx != None:
        
        if order_flag == "e1->e2":
            idx += 0
        else:
            idx += 1
        
        pass
    else:
        idx = 0
    return idx


def get_enriched_vec(args, E_trigger, Trigger2vec):
    E_closest_trigger = find_closest_trigger(E_trigger, Trigger2vec)
    #E_closest_trigger = find_closest_trigger_stem(E_trigger, Trigger2vec)
    if E_closest_trigger in Trigger2vec:
        E_children_vec = Trigger2vec[E_closest_trigger]
    else:
        E_children_vec = np.random.uniform(low=-0.01,high=0.01, size=(args.children_vec_size))

    return E_closest_trigger, E_children_vec

def get_knowledge_vec(eventpair, trigger_pair2score, C_trigger_pair2score):
    trigger_pair = eventpair.event1_trigger.lower() + " " + eventpair.event2_trigger.lower()
    R_trigger_pair = eventpair.event2_trigger.lower() + " " + eventpair.event1_trigger.lower()
    knowledge_vector = []

    
    if trigger_pair in trigger_pair2score:
        score = math.log10(trigger_pair2score[trigger_pair]+1.0)
        knowledge_vector += [score2feature(score)]
    else:
        knowledge_vector += [random.uniform(-0.01,0.01)]

    if R_trigger_pair in trigger_pair2score:
        score = math.log10(trigger_pair2score[R_trigger_pair]+1.0)
        knowledge_vector += [score2feature(score)]
    else:
        knowledge_vector += [random.uniform(-0.01,0.01)]
    

    """
    closest_trigger_pair, trigger_pair_flag = find_closest_trigger_pair(eventpair.event1_trigger.lower(), eventpair.event2_trigger.lower(), trigger_pair2score)

    if trigger_pair_flag == "trigger_pair":
        score = math.log10(trigger_pair2score[closest_trigger_pair]+1.0)
        knowledge_vector = [score2feature(score), score2feature(random.uniform(-0.01,0.01))]

    elif trigger_pair_flag == "R_trigger_pair":
        score = math.log10(trigger_pair2score[closest_trigger_pair]+1.0)
        knowledge_vector = [score2feature(random.uniform(-0.01,0.01)), score2feature(score)]
    
    else:
        knowledge_vector = [score2feature(random.uniform(-0.01,0.01)), score2feature(random.uniform(-0.01,0.01))]
    """

    
    if trigger_pair in C_trigger_pair2score:
        score = math.log10(C_trigger_pair2score[trigger_pair]+1.0)
        knowledge_vector += [score2feature(score)]
    else:
        knowledge_vector += [random.uniform(-0.01,0.01)]

    if R_trigger_pair in C_trigger_pair2score:
        score = math.log10(C_trigger_pair2score[R_trigger_pair]+1.0)
        knowledge_vector += [score2feature(score)]
    else:
        knowledge_vector += [random.uniform(-0.01,0.01)]
    

    """
    if trigger_pair in T_trigger_pair2score:
        score = math.log10(T_trigger_pair2score[trigger_pair]+1.0)
        knowledge_vector += [score2feature(score)]
    else:
        knowledge_vector += [random.uniform(-0.01,0.01)]

    if R_trigger_pair in T_trigger_pair2score:
        score = math.log10(T_trigger_pair2score[R_trigger_pair]+1.0)
        knowledge_vector += [score2feature(score)]
    else:
        knowledge_vector += [random.uniform(-0.01,0.01)]
    """
    
    return knowledge_vector


def read_trigger_pair2score(fileList):
    trigger_pair2score = {}
    for file in fileList:
        input_lines = open(file, "r")
        
        for line in input_lines:
            if not line.strip():
                continue
            eventpair = EventPair(line, -1)
            trigger_pair = eventpair.event1_trigger + " " + eventpair.event2_trigger
            
            if trigger_pair not in trigger_pair2score:
                trigger_pair2score[trigger_pair] = 0.0
            trigger_pair2score[trigger_pair] += 1.0

        input_lines.close()
    return trigger_pair2score


def get_knowledge_vec_new(eventpair, subevent_trigger_pair2score, minList, maxList):
    trigger_pair = eventpair.event1_trigger.lower() + " " + eventpair.event2_trigger.lower()
    R_trigger_pair = eventpair.event2_trigger.lower() + " " + eventpair.event1_trigger.lower()
    knowledge_vector = []

    # !!!!!!!!!!!!!!!!!!!!!!!
    #for idx, trigger_pair2score in enumerate([subevent_trigger_pair2score, temporal_trigger_pair2score, causal_trigger_pair2score]):
    for idx, trigger_pair2score in enumerate([subevent_trigger_pair2score]):
    #for trigger_pair2score in [temporal_trigger_pair2score]:
    #for trigger_pair2score in [causal_trigger_pair2score]:
        if trigger_pair in trigger_pair2score:
            
            #score = math.log10(trigger_pair2score[trigger_pair]+1.0)
            #score = math.log10(trigger_pair2score[trigger_pair]+0.1)
            score = math.log(trigger_pair2score[trigger_pair]+0.1)
            #score = math.log2(trigger_pair2score[trigger_pair]+0.1)
            knowledge_vector += [score2feature(score)]
            
            #knowledge_vector += [(trigger_pair2score[trigger_pair]-minList[idx]+1.0) / (maxList[idx] - minList[idx])]
        else:
            knowledge_vector += [random.uniform(-0.01,0.01)]
            #knowledge_vector += [random.uniform(-0.001,0.001)]

        if R_trigger_pair in trigger_pair2score:
            
            #score = math.log10(trigger_pair2score[R_trigger_pair]+1.0)
            #score = math.log10(trigger_pair2score[R_trigger_pair]+0.1)
            score = math.log(trigger_pair2score[R_trigger_pair]+0.1)
            #score = math.log2(trigger_pair2score[R_trigger_pair]+0.1)
            knowledge_vector += [score2feature(score)]
            
            #knowledge_vector += [(trigger_pair2score[R_trigger_pair]-minList[idx]+1.0) / (maxList[idx] - minList[idx])]
        else:
            knowledge_vector += [random.uniform(-0.01,0.01)]
            #knowledge_vector += [random.uniform(-0.001,0.001)]
    
    return knowledge_vector


def get_filename2instanceList_new(args, trigger_pair2score, parentTrigger2children_vec, childTrigger2parents_vec, tokenizer):
    event2vec = read_event_vec("../run_Trans_50x_50d_news/test_emb_20.txt")

    subevent_trigger_pair2score = read_trigger_pair2score(["../subevent_pairs/all_subevent_pairs.txt"])

    minList = []
    maxList = []
    for trigger_pair2score in [subevent_trigger_pair2score]:
        scoreList = []
        for trigger_pair in trigger_pair2score:
            scoreList.append(trigger_pair2score[trigger_pair])
        minList.append(min(scoreList))
        maxList.append(max(scoreList))
    print("minList:", minList)
    print("maxList:", maxList)

    MASK_id = tokenizer._convert_token_to_id("[MASK]")
    output = open("get_filename2instanceList.log", "w", 1)


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
            if args.sentence_setting == "within" and masked_sentence1 != masked_sentence2:
                continue
            elif args.sentence_setting == "across" and masked_sentence1 == masked_sentence2:
                continue
            instance = {}

            if order_flag == "e1->e2":
                word_pair = "< " + event1 + " > " + relation + " < " + event2 + " >"
            else:
                word_pair = "< " + event2 + " > " + "R_" + relation + " < " + event1 + " >"

            eventpair = EventPair(word_pair, -1)

            
            # KE
            E1_vec = event2vec[eventpair.event1.lower()]
            E2_vec = event2vec[eventpair.event2.lower()]
            #children_vector = np.concatenate((E1_vec, E2_vec), axis=0)
            #children_vector = np.concatenate((E1_vec, E2_vec, E2_vec-E1_vec), axis=0)
            children_vector = E2_vec-E1_vec

            #knowledge_vector = get_knowledge_vec(eventpair, trigger_pair2score, C_trigger_pair2score)
            knowledge_vector = get_knowledge_vec_new(eventpair, subevent_trigger_pair2score, minList, maxList)
            
            #print("word_pair:", word_pair)
            #print("closest_trigger_pair:", closest_trigger_pair)
            #print("knowledge_vector:", knowledge_vector)
            output.write("word_pair: " + str(word_pair) + "\n")
            #output.write("closest_trigger_pair: " + str(closest_trigger_pair) + "\n")
            output.write("knowledge_vector: " + str(knowledge_vector) + "\n\n")
            #print(relation, get_class_idx(args, relation, order_flag))
            #print("\n")

            
            if masked_sentence1 == masked_sentence2:
                masked_sentence = "[CLS] " + masked_sentence1 + " [SEP]"
            elif order_flag == "e1->e2":
                masked_sentence = "[CLS] " + masked_sentence1 + " [SEP] " + masked_sentence2 + " [SEP]"
            elif order_flag == "e2->e1":
                masked_sentence = "[CLS] " + masked_sentence2 + " [SEP] " + masked_sentence1 + " [SEP]"

            input_ids = tokenizer.encode(masked_sentence)
            input_ids, input_mask = seq_padding(args, input_ids)

            masked_idxList = []

            for i, input_id in enumerate(input_ids):
                if input_id == MASK_id:
                    masked_idxList.append(i)
            if len(masked_idxList) != 2:
                print("Type1: One test instance exceeds max_seq_length.")
                continue
            
            if args.mask_trigger == True:
                instance = {"event_pair": word_pair, "masked_sentence": masked_sentence, "input_ids": input_ids, 
                            "masked_idxList": masked_idxList, "knowledge_vector": knowledge_vector, 
                            "children_vector": children_vector}
            
            else:
                if sentence1 == sentence2:
                    sentence = "[CLS] " + sentence1 + " [SEP]"
                elif order_flag == "e1->e2":
                    sentence = "[CLS] " + sentence1 + " [SEP] " + sentence2 + " [SEP]"
                elif order_flag == "e2->e1":
                    sentence = "[CLS] " + sentence2 + " [SEP] " + sentence1 + " [SEP]"

                input_ids = tokenizer.encode(sentence)
                input_ids, input_mask = seq_padding(args, input_ids)

                trigger1_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event1_trigger)]
                trigger2_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.tokenize(event2_trigger)]

                if order_flag == "e1->e2":
                    masked_idxList[1] = masked_idxList[1] + (len(trigger1_ids) - 1) # How many slots shift to the right
                else:
                    masked_idxList[1] = masked_idxList[1] + (len(trigger2_ids) - 1) # How many slots shift to the right
                
                if masked_idxList[1] >= len(input_ids):
                    print("Type2: One test instance exceeds max_seq_length.")
                    continue

                instance = {"event_pair": word_pair, "masked_sentence": masked_sentence, "input_ids": input_ids, 
                            "masked_idxList": masked_idxList, "knowledge_vector": knowledge_vector, 
                            "children_vector": children_vector}

            # !!!!!!!!!!!!!!!!!!!!!
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
    output.close()

    return filename2instanceList


def prepare_data(filename2instanceList, test_files):
    
    trainList = []
    devList = []
    testList = []

    filenames = list(filename2instanceList.keys())
    
    for filename in filenames:
        if filename in test_files:
            testList += filename2instanceList[filename]
        else:
            trainList += filename2instanceList[filename]


    print("trainList[:5]", trainList[:5])
    print("devList[:5]", devList[:5])
    print("testList[:5]", testList[:5])
    env = {}
    env["train"] = trainList
    env["dev"] = devList
    env["test"] = testList
    pickle.dump(env, open("env.pkl", "wb"))

    
    
    