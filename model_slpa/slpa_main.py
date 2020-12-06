import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
from nltk.corpus import wordnet as wn
import ast, time, os, math, copy, gensim, glob, nltk, pdb, argparse
import cPickle as pickle
from multiprocessing import Process
from slpa_utilities import Graph, MetaGraph, get_acceptedLabel, initialize_memory, find_communities, graph_density
from utilities import EventPair, extract_valid_pairs, extract_BERT_predicted_pairs, remove_brackets, light_verbs, pronouns, person_pronouns

model = gensim.models.KeyedVectors.load_word2vec_format("../../tools/GoogleNews-vectors-negative300.bin", binary = True)


B_light_verbs = set(["["+v+"]" for v in light_verbs])

def cal_wordList_sim(args, wordList1, wordList2, invalid_words):
    simList = []
    for word1 in wordList1:
        if word1 in invalid_words:
            continue
        for word2 in wordList2:
            if word2 in invalid_words:
                continue
            if word1 == word2:
                #simList.append(float(len(word1))/6.0)
                simList.append(1.0)
            #elif word1 in model and word2 in model:
            elif word1+" "+word2 in args.word_pair2sim:
                #simList.append(model.similarity(word1, word2))
                simList.append(args.word_pair2sim[word1+" "+word2])
            else:
                simList.append(0.0)
    if len(simList) != 0:
        sim_avg = sum(simList) / float(len(simList))
    else:
        sim_avg = 0
    return sim_avg

THRESHOLD = 0.32

def measure_sim(args, E1, E2, seed_pairs, invalid_words, word2definition_words, word2synonyms, type2w):
    W = 0
    typeList = []

    if "wn_" in E1 and "wn_" in E2:
        E1_trigger = E1.split("_")[1]
        idx = int(E1.split("_")[2])
        E1_definition_words = word2definition_words[E1_trigger][idx]

        E2_trigger = E2.split("_")[1]
        idx = int(E2.split("_")[2])
        E2_definition_words = word2definition_words[E2_trigger][idx]

        sim = cal_wordList_sim(args, E1_definition_words, E2_definition_words, invalid_words)
        if sim >= 0.3:
            W += type2w["DF_w"] * sim
            typeList.append("DF_w")
            return W, typeList
        else:
            return None, typeList

    elif "wn_" in E1:
        E1_trigger = E1.split("_")[1]
        idx = int(E1.split("_")[2])

        E1_trigger_event_phrase = "< [" + E1_trigger + "] >"
        if E1_trigger_event_phrase + " -> " + E2 in seed_pairs:
            W += type2w["SEP_w"]
            typeList.append("SEP_w")

        definition_words = word2definition_words[E1_trigger][idx]
        E2_words = remove_brackets(E2).split()
        sim = cal_wordList_sim(args, definition_words, E2_words, invalid_words)
    
        if sim >= THRESHOLD:
            W += type2w["DF_w"] * sim
            typeList.append("DF_w")

        for word in E2.split():
            if word[0] == "[":
                E2_trigger = word.replace("[","").replace("]","")

        if E1_trigger not in light_verbs:
            synonyms = word2synonyms[E1_trigger][idx]
            if E1_trigger != E2_trigger and E2_trigger in synonyms:
                W += type2w["SYN_w"]
                typeList.append("SYN_w")

        if W == 0:
            return None, typeList
        else:
            return W, typeList

    elif "wn_" in E2:
        E2_trigger = E2.split("_")[1]
        idx = int(E2.split("_")[2])

        E2_trigger_event_phrase = "< [" + E2_trigger + "] >"
        if E2_trigger_event_phrase + " -> " + E1 in seed_pairs:
            W += type2w["SEP_w"]
            typeList.append("SEP_w")

        definition_words = word2definition_words[E2_trigger][idx]
        E1_words = remove_brackets(E1).split()
        sim = cal_wordList_sim(args, definition_words, E1_words, invalid_words)
        
        if sim >= THRESHOLD:
            W += type2w["DF_w"] * sim
            typeList.append("DF_w")

        for word in E1.split():
            if word[0] == "[":
                E1_trigger = word.replace("[","").replace("]","")

        if E2_trigger not in light_verbs:
            synonyms = word2synonyms[E2_trigger][idx]
            if E1_trigger != E2_trigger and E1_trigger in synonyms:
                W += type2w["SYN_w"]
                typeList.append("SYN_w")

        if W == 0:
            return None, typeList
        else:
            return W, typeList

    else:

        E1_args = []
        for E1_word in set(E1.split())-set(["<",">"])-invalid_words:
            if E1_word[0] != "[":
                E1_args.append(E1_word)
            else:
                E1_trigger = E1_word.replace("[","").replace("]","")

        E2_args = []
        for E2_word in set(E2.split())-set(["<",">"])-invalid_words:
            if E2_word[0] != "[":
                E2_args.append(E2_word)
            else:
                E2_trigger = E2_word.replace("[","").replace("]","")

        if len(E1_args) != 0 and len(E2_args) != 0:
            sim = cal_wordList_sim(args, E1_args, E2_args, invalid_words)
            if sim >= THRESHOLD:
                W += type2w["EM_w"] * sim
                typeList.append("EM_w_arguments")
        
        if (E1_trigger not in light_verbs) and (E2_trigger not in light_verbs):
            if E1_trigger == E2_trigger:
                #sim = float(len(E1_trigger))/6.0
                sim = 1.0
                W += type2w["EM_w"] * sim
                typeList.append("EM_w_trigger")
            elif E1_trigger+" "+E2_trigger in args.word_pair2sim:
                #sim = model.similarity(E1_trigger, E2_trigger)
                sim = args.word_pair2sim[E1_trigger+" "+E2_trigger]
                if sim >= 0.4:
                    W += type2w["EM_w"] * sim
                    typeList.append("EM_w_trigger")

        if (E1_trigger not in light_verbs) and len(E2_args) != 0:
            sim = cal_wordList_sim(args, [E1_trigger], E2_args, invalid_words)
            if sim >= THRESHOLD:
                W += type2w["EM_w"] * sim
                typeList.append("EM_w_TriggerArgs")

        if (E2_trigger not in light_verbs) and len(E1_args) != 0:
            sim = cal_wordList_sim(args, [E2_trigger], E1_args, invalid_words)
            if sim >= THRESHOLD:
                W += type2w["EM_w"] * sim
                typeList.append("EM_w_TriggerArgs")

        if (E1 + " -> " + E2 in seed_pairs) or (E2 + " -> " + E1 in seed_pairs):
            W += type2w["SEP_w"]
            typeList.append("SEP_w")
        
        if W == 0:
            return None, typeList
        else:
            return W, typeList

# get all neighbors' similarity score for tweet i (multiprocessing)
def get_neighbors_sim(args, idx, seed_pairs, all_events, invalid_words, word2definition_words, word2synonyms, type2w):
    ij2sim = {}
    output = open(str(idx) + "_event_pairs.txt", "w", 10)
    out_count = 0

    for i in tqdm(range(0, len(all_events))):
        if i % 20 != idx:
            continue
        E1 = all_events[i]
        for j in range(i+1, len(all_events)):
            E2 = all_events[j]

            sim, typeList = measure_sim(args, E1, E2, seed_pairs, invalid_words, word2definition_words, word2synonyms, type2w)

            if sim != None:
                ij2sim[str(i) + ' ' + str(j)] = sim
            if out_count < 20000 and sim != None and random.uniform(0, 1) < 0.02:
                out_count += 1
                output.write("############################\n")
                output.write(str(E1) + "\n")
                output.write(str(E2) + "\n")
                output.write("****************************\n")
                output.write(str(sim) + " " + str(typeList) + "\n")
                output.write("############################\n\n\n")
    pickle.dump(ij2sim, open("ij2sim_" + str(idx) + ".p", "wb"))
    output.close()


def create_semantic_graph(args, seed_pairs, all_events, invalid_words, word2definition_words, word2synonyms, type2w):
    G = Graph()
    print("building semantic graph...")

    
    # for any two events, build the connection between them
    processV = []
    for idx in range(0, 20):
        processV.append(Process(target = get_neighbors_sim, 
            args = (args, idx, seed_pairs, all_events, invalid_words, word2definition_words, word2synonyms, type2w, )))
    for idx in range(0, 20):
        processV[idx].start()
    for idx in range(0, 20):
        processV[idx].join()
    

    for idx in range(0, 20):
    #for idx in range(0, 2):
        print("ij2sim_" + str(idx) + ".p")
        ij2sim = pickle.load(open("ij2sim_" + str(idx) + ".p", "rb"))
        for ij in ij2sim:
            sim = ij2sim[ij]
            #if sim != None and sim != 0 and sim >= 0.19:
            if sim != None and sim != 0:
                i = int(ij.split()[0])
                j = int(ij.split()[1])
                
                G.add_edge(i, j, sim)
                G.add_edge(j, i, sim)
        #os.system("rm " + "ij2sim_" + str(idx) + ".p")

    print("graph_density(G):", graph_density(G))

    return G

def extract_meaningful_words(definition):
    text = nltk.word_tokenize(definition)
    meaningful_words = set()
    for item in nltk.pos_tag(text):
        if item[1][0] in ["N", "V"]:
            meaningful_words.add(item[0])
    return meaningful_words

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    invalid_words = set(["person", "location", "be", "time", "act", "event", "activities",
                         "of", "for", "to", "up", "on", "with", "not", "at", "from", "into", "over", "by", "against","poss",
                         "about", "off", "before"])
    invalid_words = invalid_words | light_verbs | pronouns | person_pronouns
    
    random.seed(11)
    seed_pairs = extract_valid_pairs("../run_extract_event_pair_nmod2/news/sorted_parent_child2num.txt", 2.0)
    

    #seed_pairs = set(list(seed_pairs)[:8000])

    vocab = []

    all_parents = set()
    all_children = set()
    for pair in seed_pairs:
        eventpair = EventPair(pair, -1)
        all_parents.add(eventpair.event1)
        all_children.add(eventpair.event2)

    
    for pair in seed_pairs:
        words = pair.split()
        for w in words:
            if w in ['<', '>']:
                continue
            vocab.append(w.replace("[", "").replace("]", ""))

    
    all_parent_triggers = set()
    for pair in seed_pairs:
        eventpair = EventPair(pair, -1)
        all_parent_triggers.add(eventpair.event1_trigger)

    word2definition_words = {}
    word2synonyms = {}
    for event in all_parent_triggers:
        word = remove_brackets(event)
        synsets = wn.synsets(word, pos='n')
        if len(synsets) != 0:
            word2definition_words[word] = []
            word2synonyms[word] = []
            for i in range(0, min(len(synsets), 5)):
                definition = wn.synsets(word, pos='n')[i].definition()
                definition_words = extract_meaningful_words(definition)
                word2definition_words[word] += [definition_words]
                synonyms = set()
                syn = wn.synsets(word, pos='n')[i]
                for l in syn.lemmas():
                    synonyms.add(l.name())
                word2synonyms[word] += [synonyms]
                vocab += list(definition_words)
                vocab += list(synonyms)

    wn_events = []
    for word in word2definition_words:
        definition_num = len(word2definition_words[word])
        for i in range(definition_num):
            wn_events.append("wn_"+word+"_"+str(i)+"_max"+str(definition_num))
    
    # for word1 in word2definition_words:
    #     for word2 in word2definition_words:
    #         if word2 in word2definition_words[word1][0]:
    #             print(word1, word2, word2definition_words[word1])
    #             pdb.set_trace()

    
    word_pair2sim = {}
    vocab = list(set(vocab))
    print("len(vocab):", len(vocab))
    for i in tqdm(range(0, len(vocab))):
        for j in range(i+1, len(vocab)):
            if vocab[i] in model and vocab[j] in model:
                sim = model.similarity(vocab[i], vocab[j])
                #if sim >= 0.25:
                if sim >= 0.20:
                    word_pair2sim[vocab[i] + " " + vocab[j]] = sim
                    word_pair2sim[vocab[j] + " " + vocab[i]] = sim
    pickle.dump(word_pair2sim, open("word_pair2sim.p", "wb"))
    
    print("load word_pair2sim.p...")
    word_pair2sim = pickle.load(open("word_pair2sim.p", "rb"))
    args.word_pair2sim = word_pair2sim
    

    
    all_events = all_parents | all_children
    all_events = list(all_events)
    all_events = all_events + wn_events
    
    event_str2idx = {event_str:idx for idx,event_str in enumerate(all_events)}
    idx2event_str = {idx:event_str for idx,event_str in enumerate(all_events)}

    ################################ Part I ##################################
    # wordnet_w, common_words_w, subevent_pair_w, definition_w, synonym_w, wordnet_same_word
    # "CW_w":0.05,
    #type2w = {"EM_w":0.5, "SEP_w":0.5, "DF_w":2.5, "SYN_w":2.5}
    type2w = {"EM_w":0.5, "SEP_w":0.5, "DF_w":0.5, "SYN_w":0.5}
    event_idx2labels = prepare_seed_events(all_events)
    G = create_semantic_graph(args, seed_pairs, all_events, invalid_words, word2definition_words, word2synonyms, type2w)
    
    T = 60
    r = 0.30
    #decay_r = 0.95
    decay_r = 1.0

    event_str2labels = {}
    print("running community detection (pass 1)...")
    memory = initialize_memory(G, idx2event_str, event_idx2labels, "1")
    communities = find_communities(G, T, r, decay_r, memory)
    pickle.dump(communities, open("communities_pass1.p", "wb"))

    communitiesList = pickle.load(open("communitiesList.p", "rb"))
    memoryList = pickle.load(open("memoryList.p", "rb"))

    ################################ Part II #################################
    last_communities_idx = len(communitiesList) - 1
    communities = communitiesList[-1]
    output_folder = "communities_" + str(last_communities_idx) + "/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.system("rm " + output_folder + "*")
    count = 0
    for key in communities:
        count += 1
        output = open(output_folder + str(count) + "_" + str(key) + ".txt", "w")
        
        for idx in communities[key]:
            if type(idx) == unicode or type(idx) == str:
                continue
            output.write(str(idx2event_str[idx]) + "\n")
            output.write(str(memoryList[last_communities_idx][idx]) + "\n\n")
            
        output.close()
    
    output_folder = "communities_" + str(last_communities_idx) + "/"
    ################################ Part III ################################
    output = open("clusters_top_words.txt", "w")
    for file in glob.glob(output_folder + "*.txt"):
        input_lines = open(file, "r")
        word2freq = {}
        event_count = 0
        for line in input_lines:
            if not line.strip():
                continue
            if line[0] != "{":
                event_count += 1
                if "wn_" in line:
                    words = line.split("_")
                    w = "wn_"+words[1]
                    if w not in word2freq:
                        word2freq[w] = 0
                    word2freq[w] += 1
                else:
                    words = line.split()
                    for w in words:
                        if w in ["<", ">"]:
                            continue
                        if w not in word2freq:
                            word2freq[w] = 0
                        word2freq[w] += 1
        for word in word2freq:
            #word2freq[word] = [word2freq[word], float(word2freq[word])/float(event_count)]
            word2freq[word] = [word2freq[word]]
        sorted_word2freq = sorted(word2freq.items(), key=lambda e: e[1][0], reverse = True)
        valid_items = []
        for item in sorted_word2freq:
            if item[1][0] >= 2 and item[0] not in invalid_words:
                valid_items.append(item)
        if len(valid_items) >= 3:
            output.write(file+"\n")
            output.write(str(valid_items)+"\n\n\n")

        input_lines.close()
    output.close()
    

    output_folder = "communities_" + str(last_communities_idx) + "/"
    event2cluster_ids = {}
    ################################ Part IV ################################
    print("############ Part IV ############")
    for file in glob.glob(output_folder + "*.txt"):
        input_lines = open(file, "r")
        cluster_id = file.split("/")[-1]
        for line in input_lines:
            if not line.strip():
                continue
            if "wn_" in line:
                words = line.split("_")
                event = words[0]+"_"+words[1]
                trigger_event = "< [" + words[1] + "] >"
            elif "<" in line:
                words = line.split()
                event = " ".join(words)
                for w in words:
                    if w[0] == "[":
                        trigger_event = "< " + w + " >"

            if event not in event2cluster_ids:
                event2cluster_ids[event] = set()
            if trigger_event not in event2cluster_ids and trigger_event in all_parents:
                event2cluster_ids[trigger_event] = set()
            event2cluster_ids[event].add(cluster_id)
            if trigger_event in all_parents:
                event2cluster_ids[trigger_event].add(cluster_id)
    
    valid_output_pairs = set()
    invalid_output_pairs = set()
    other_output_pairs = set()
    for seed_pair in seed_pairs:
        eventpair = EventPair(seed_pair, -1)
        
        if eventpair.event2 not in event2cluster_ids:
            other_output_pairs.add(seed_pair)
            continue
        child_cluster_ids = event2cluster_ids[eventpair.event2]
        parent_cluster_ids = set()
        if eventpair.event1 in event2cluster_ids:
            parent_cluster_ids = parent_cluster_ids | event2cluster_ids[eventpair.event1]
        if "wn_"+eventpair.event1_trigger in event2cluster_ids:
            parent_cluster_ids = parent_cluster_ids | event2cluster_ids["wn_"+eventpair.event1_trigger]
        if "< " + eventpair.event1_trigger + " >" in event2cluster_ids:
            parent_cluster_ids = parent_cluster_ids | event2cluster_ids["< " + eventpair.event1_trigger + " >"]

        if len(parent_cluster_ids) == 0:
            other_output_pairs.add(seed_pair)
            continue
        if len(child_cluster_ids & parent_cluster_ids) != 0:
            #print("valid:", seed_pair)
            valid_output_pairs.add(seed_pair)
        else:
            #print("invalid:", seed_pair)
            invalid_output_pairs.add(seed_pair)

        #raw_input("continue")
    valid_output = open("valid_pairs.txt", "w")
    invalid_output = open("invalid_pairs.txt", "w")
    other_output = open("other_pairs.txt", "w")
    input_lines = open("../run_extract_event_pair_nmod2/news/sorted_parent_child2num.txt", "r")
    for line in input_lines:
        pair = line.split(" | ")[0]
        if pair in valid_output_pairs:
            valid_output.write(line)
            valid_output_pairs.remove(pair)
        elif pair in invalid_output_pairs:
            invalid_output.write(line)
            invalid_output_pairs.remove(pair)
        elif pair in other_output_pairs:
            other_output.write(line)

    for pair in valid_output_pairs:
        valid_output.write(pair + "\n")

    for pair in invalid_output_pairs:
        invalid_output.write(pair + "\n")
    valid_output.close()
    invalid_output.close()
    other_output.close()

