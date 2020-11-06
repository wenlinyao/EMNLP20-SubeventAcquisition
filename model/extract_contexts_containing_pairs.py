import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import glob, math, pickle, random, ast
import numpy as np
from multiprocessing import Process
from utilities import EventPair, load_not_event
from utilities import document2sentenceList, detect_event_trigger, old_extract_event_with_arg, get_path, get_path2, extract_valid_pairs
from utilities import extract_candidates_conj, extract_candidates_surrounding, extract_CT_pairs

# extract contexts for all possible pairs
# split those contexts into train and candidate in BERT_context_model_main.py
def extract_all_pairs(file): 
    all_pairs = set()

    input_file = open(file, "r")
    
    for line in input_file:
        items = line.split(" | ")
        pair = items[0]
        score = float(items[1])
        freqList = ast.literal_eval(items[2])
        if score >= 1.1 or freqList[0] >= 1:
            all_pairs.add(pair)

    input_file.close()
    return all_pairs



def extract_contexts(sentenceList, all_pairs, CT_pairs):

    pos_contexts = []
    neg_contexts = []
    CT_contexts = []

    
    for sentence in sentenceList:
        sentence_str = " ".join(sentence.wordList)
        for i in range(1, len(sentence.wordList)-2):
            if sentence.lemmaList[i] == "say":
                continue
            if detect_event_trigger(sentence, i) != True:
                continue
            
            index_result1, words_result1, tag1 = old_extract_event_with_arg(sentence, i)

            for j in range(i+2, len(sentence.wordList)):
                if sentence.lemmaList[j] == "say":
                    continue
                if detect_event_trigger(sentence, j) != True:
                    continue
                
                index_result2, words_result2, tag2 = old_extract_event_with_arg(sentence, j)

                if tag1 == "verb_alone" and tag2 == "verb_alone":
                    continue
                
                event1 = "< " + " ".join(words_result1).lower() + " >"
                event2 = "< " + " ".join(words_result2).lower() + " >"

                word_pair1 = event1 + " -> " + event2
                word_pair2 = event2 + " -> " + event1

                CT_word_pair1 = event1 + " => " + event2
                CT_word_pair2 = event2 + " => " + event1
                
                for index in index_result1:
                    if index[0] == "[":
                        mask1_idx = int(index.replace("[","").replace("]",""))
                for index in index_result2:
                    if index[0] == "[":
                        mask2_idx = int(index.replace("[","").replace("]",""))
                
                masked_sentence = " ".join([sentence.wordList[k] if k not in [mask1_idx, mask2_idx] else "[MASK]" for k in range(1, len(sentence.wordList))])  

                if word_pair1 in all_pairs:
                    valid_index, word_path, path_pos = get_path(sentence, str(i), str(j))
                    pos_contexts.append({"sentence": " ".join(sentence.wordList[1:]), "masked_sentence": masked_sentence, "word_path": word_path, "word_pair": word_pair1, "trigger_index": str(i)+" "+str(j)})
                    

                elif word_pair2 in all_pairs:
                    valid_index, word_path, path_pos = get_path(sentence, str(i), str(j))
                    pos_contexts.append({"sentence": " ".join(sentence.wordList[1:]), "masked_sentence": masked_sentence, "word_path": word_path, "word_pair": event1 + " <- " + event2, "trigger_index": str(i)+" "+str(j)})
                

                elif CT_word_pair1 in CT_pairs:
                    valid_index, word_path, path_pos = get_path(sentence, str(i), str(j))
                    CT_contexts.append({"sentence": " ".join(sentence.wordList[1:]), "masked_sentence": masked_sentence, "word_path": word_path, "word_pair": CT_word_pair1, "trigger_index": str(i)+" "+str(j)})
                
                elif CT_word_pair2 in CT_pairs:
                    valid_index, word_path, path_pos = get_path(sentence, str(i), str(j))
                    CT_contexts.append({"sentence": " ".join(sentence.wordList[1:]), "masked_sentence": masked_sentence, "word_path": word_path, "word_pair": event1 + " <= " + event2, "trigger_index": str(i)+" "+str(j)})
                
                elif random.uniform(0, 1) < 0.05:
                    valid_index, word_path, path_pos = get_path(sentence, str(i), str(j))
                    #if valid_index != None:
                    neg_contexts.append({"sentence": " ".join(sentence.wordList[1:]), "masked_sentence": masked_sentence, "word_path": word_path, "word_pair": event1 + " <-> " + event2, "trigger_index": str(i)+" "+str(j)})
                

    return pos_contexts, neg_contexts, CT_contexts

def process_document(folder, newswire, idx, output_dir, all_pairs, CT_pairs):
    count = 0

    sentence_str_set = set()

    for file_name in glob.glob(folder + "new_" + newswire + "_eng_*" + idx + ".txt"):
        pos_found_contexts = []
        neg_found_contexts = []
        CT_found_contexts = []

        count += 1
        #if count > 1:
        #    break
        print file_name

        input_file = open(file_name, "r")
        
        document = []
        doc_id = ""
        doc_type = ""
        DOC_flag = False
        for line in input_file:
            if not line.strip():
                continue
            words = line.split()
            if words[0] == "<DOC>":
                document = []
                doc_id = words[1]
                doc_type = words[2]
                DOC_flag = True
                continue
            if words[0] == "</DOC>":
                DOC_flag = False
                if doc_type == "story":
                    sentenceList, paragraph_boundary, coreferenceList = document2sentenceList(document)
                    pos_contexts, neg_contexts, CT_contexts = extract_contexts(sentenceList, all_pairs, CT_pairs)
                    pos_found_contexts += pos_contexts
                    neg_found_contexts += neg_contexts
                    CT_found_contexts += CT_contexts

                continue
            if DOC_flag == True:
                document.append(line)
                continue
        input_file.close()

        random.shuffle(pos_found_contexts)
        random.shuffle(neg_found_contexts)
        random.shuffle(CT_found_contexts)

        output = open(output_dir + file_name.split("/")[-1].replace(".txt", "_pos_found_contexts.txt"), "w")

        pair2freq = {}
        for context in pos_found_contexts:
            if context["sentence"] in sentence_str_set:
                continue
            else:
                sentence_str_set.add(context["sentence"])

            if context["word_pair"] not in pair2freq:
                pair2freq[context["word_pair"]] = 1
            elif pair2freq[context["word_pair"]] < 10:
                pair2freq[context["word_pair"]] += 1
            else:
                continue

            output.write("<word_pair> " + context["word_pair"] + "\n")
            output.write("<trigger_index> " + context["trigger_index"] + "\n")
            output.write("<word_path> " + str(context["word_path"]) + "\n")
            output.write("<masked_sentence> "+ context["masked_sentence"] + "\n")
            output.write("<sentence> "+ context["sentence"] + "\n\n")

        output.close()

        output = open(output_dir + file_name.split("/")[-1].replace(".txt", "_neg_found_contexts.txt"), "w")

        for context in neg_found_contexts:
            if context["sentence"] in sentence_str_set:
                continue
            else:
                sentence_str_set.add(context["sentence"])
            output.write("<word_pair> " + context["word_pair"] + "\n")
            output.write("<trigger_index> " + context["trigger_index"] + "\n")
            output.write("<word_path> " + str(context["word_path"]) + "\n")
            output.write("<masked_sentence> "+ context["masked_sentence"] + "\n")
            output.write("<sentence> "+ context["sentence"] + "\n\n")

        output.close()

        # # historical issue (use TC)
        output = open(output_dir + file_name.split("/")[-1].replace(".txt", "_TC_found_contexts.txt"), "w")

        for context in CT_found_contexts:
            if context["sentence"] in sentence_str_set:
                continue
            else:
                sentence_str_set.add(context["sentence"])
            output.write("<word_pair> " + context["word_pair"] + "\n")
            output.write("<trigger_index> " + context["trigger_index"] + "\n")
            output.write("<word_path> " + str(context["word_path"]) + "\n")
            output.write("<masked_sentence> "+ context["masked_sentence"] + "\n")
            output.write("<sentence> "+ context["sentence"] + "\n\n")

        output.close()

        


if __name__ == "__main__":
    folder = "../preprocess_gigaword/new_run/"
    #newswire_pool = ["nyt"]
    #newswire_pool = ["xin"]
    newswire_pool = ["nyt", "afp", "apw", "cna", "ltw", "wpb", "xin"]
    output_dir = "news/"

    #seed_pairs, candidate_pairs = extract_seed_cand_pairs("../run_extract_event_pair_nmod2/news/sorted_parent_child2num.txt", 2.0)
    #seed_pairs, candidate_pairs = extract_seed_cand_pairs("../run_extract_event_pair_nmod2/news/selected_pairs.txt", 2.0)
    all_pairs = extract_all_pairs("../run_extract_event_pair_nmod2/news/sorted_parent_child2num.txt")

    candidate_pairs_conj = extract_candidates_conj("../run_extract_event_pair_conj2/news/sorted_parent_child2num.txt", 1.0)
    candidate_pairs_surrounding = extract_candidates_surrounding("../run_extract_surrounding_subevents/extract_subevent_pairs.txt", 1.0)
    news_C_pairs, news_T_pairs = extract_CT_pairs("../BBN_project/run_extract_pairs_patterns_small/sorted_event_pairs.txt", 2.0)
    CT_pairs = news_C_pairs | news_T_pairs

    print("nmod pairs:", len(all_pairs))
    print("conj pairs:", len(candidate_pairs_conj))
    print("surrounding pairs:", len(candidate_pairs_surrounding))
    print("Causal Temporal pairs:", len(CT_pairs))

    all_pairs = all_pairs | candidate_pairs_conj | candidate_pairs_surrounding

    
    print ("len(all_pairs):", len(all_pairs))
    print (list(all_pairs)[:30])

    for newswire in newswire_pool:
        print newswire
        processV = []
        for idx in range(0, 10):
            processV.append(Process(target = process_document, args = (folder, newswire, str(idx), output_dir, all_pairs, CT_pairs,)))
        for idx in range(0, 10):
            processV[idx].start()
            
        for idx in range(0, 10):
            processV[idx].join()

        print newswire, "finished!"
    
