import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import glob, math, pickle, argparse
import numpy as np
from utilities import EventPair, validate_eventpair, clean_eventpair
from model_utils import non_zero_count

def entropy(pList):
    normalize_pList = []
    for p in pList:
        normalize_pList.append(p / sum(pList))
    ent = 0
    for p in normalize_pList:
        if p != 0:
            ent -= p * math.log(p, 2)
    return ent

def sum_root(numList):
    return sum([num ** (1.0/5.0) for num in numList])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", dest="folder", type=str, metavar='<str>', default='../run_extract_sentences_nmod2/', help="Where to find extracted sentences.")
    parser.add_argument("--genre", dest="genre", type=str, metavar='<str>', default='news/', help="Run event pair extraction on which genre.")
    args = parser.parse_args()


    if "nmod" in args.folder:
        relationList = ["nmod:during", "nmod:in", "nmod:amid", "nmod:throughout", "nmod:including", "nmod:within"]
        #relationList = ["nmod:during",             "nmod:amid", "nmod:throughout", "nmod:including", "nmod:within"]
    elif "advcl" in args.folder:
        relationList = ["advcl:while", "advcl:when"]

    parent_child2num = {}

    doc_id2pairs = {}
    
    
    #for folder in ["news/", "novel/"]:
    #for folder in ["news/"]:
    #for folder in ["novel/"]:

    sentence_str_set = set()

    
    for file in glob.glob(args.folder + args.genre + "*.txt"):
        print(file)
        input_file = open(file, "r")

        for line in input_file:
            if not line.strip():
                continue
            
            words = line.split()
            if words[0] == "<doc_id>":
                doc_id = words[1]
                continue

            if words[0] == "<subevent>":
                eventpair = EventPair(" ".join(words[1:]), 1)

                if "including" in eventpair.relation:
                    pair = eventpair.event1 + " -> " + eventpair.event2
                else:
                    pair = eventpair.event2 + " -> " + eventpair.event1

                pair = clean_eventpair(pair)
                continue

            if words[0] == "<word>":
                sentence_str = " ".join(words)
                if sentence_str in sentence_str_set:
                    continue
                else:
                    sentence_str_set.add(sentence_str)
                
                if validate_eventpair(eventpair) == False:
                    continue

                if eventpair.relation not in relationList: # Ignore nmod:in?
                    continue

                if pair not in parent_child2num:
                    parent_child2num[pair] = [0.0 for i in range(0, len(relationList)+1)]

                if doc_id not in doc_id2pairs:
                    doc_id2pairs[doc_id] = set()
                doc_id2pairs[doc_id].add(pair)


                parent_child2num[pair][relationList.index(eventpair.relation)] += 1
                parent_child2num[pair][-1] += 1


        input_file.close()

    pickle.dump(parent_child2num, open(args.genre + "parent_child2num.p", "wb"))
    #pickle.dump(doc_id2pairs, open(args.genre + "doc_id2pairs.p", "wb"))
    


    print ("loading parent_child2num.p")
    parent_child2num = pickle.load(open(args.genre + "parent_child2num.p", "rb"))
    parent_child2score = {}

    for parent_child in parent_child2num:
        numList = parent_child2num[parent_child]
        #parent_child2score[parent_child] = (entropy(numList[:-1]) + 0.01) * math.log(numList[-1], 2)
        #parent_child2score[parent_child] = non_zero_count(numList[:-1]) * 1000 + math.log(numList[-1], 2)
        parent_child2score[parent_child] = sum_root(numList[:-1])

    sorted_parent_child2score = sorted(parent_child2score.items(), key=lambda e: e[1], reverse = True)

    output = open(args.genre + "sorted_parent_child2num.txt", "w")
    for item in sorted_parent_child2score:
        #if item[1][-1] <= 1:
        #    continue
        output.write(item[0] + " | " + str(item[1]) + " | " + str(parent_child2num[item[0]]) + "\n")
    output.close()

    output = open(args.genre + "valid_parent_child2num.txt", "w")
    for item in sorted_parent_child2score:
        numList = parent_child2num[item[0]]
        #if numList[-1] <= 3:
        #    continue
        if non_zero_count(numList[:-1]) <= 1:
            continue
        if item[1] <= 3:
            continue
        output.write(item[0] + " | " + str(item[1]) + " | " + str(parent_child2num[item[0]]) + "\n")
    output.close()

    

    parent2children = {}
    child2parents = {}

    print("len(parent_child2num): ", len(parent_child2num))

    for item in sorted_parent_child2score:
    #for parent_child in parent_child2num:
        parent_child = item[0]
        if parent_child2num[parent_child][-1] <= 1:
            continue
        parent = parent_child.split(" -> ")[0]
        child = parent_child.split(" -> ")[1]
        if parent not in parent2children:
            parent2children[parent] = [child]
        else:
            parent2children[parent] += [child]
        if child not in child2parents:
            child2parents[child] = [parent]
        else:
            child2parents[child] += [parent]

    parent_children2avg = {}
    #parent_children2countList = {}

    print("len(parent2children): ", len(parent2children))
    print("len(child2parents): ", len(child2parents))

    for parent in parent2children:
        if len(parent2children[parent]) <= 1:
            continue

        scoreList = []
        for child in parent2children[parent]:
            scoreList.append(parent_child2score[parent + " -> " + child])

        parent_children = parent + ": " + "|".join(list(parent2children[parent]))
        parent_children2avg[parent_children] = sum(scoreList) / float(len(scoreList))

    sorted_parent_children2avg = sorted(parent_children2avg.items(), key = lambda e:e[1], reverse = True)

    output = open(args.genre + "sorted_parent_children2avg.txt", "w")
    for item in sorted_parent_children2avg:
        output.write(str(item[1]) + "\n")
        parent = item[0].split(": ")[0]
        children = item[0].split(": ")[1]
        #output.write(parent + " | " + str(parent_children2countList[item[0]]) + "\n")
        output.write(parent + " | " + str(item[1]) + "\n")
        for e in children.split("|"):
            output.write(" -> " + e + " | " + str(parent_child2num[parent + " -> " + e]) + "\n")
        output.write("\n\n")
    output.close()

    child_parents2avg = {}
    for child in child2parents:
        if len(child2parents[child]) <= 1:
            continue

        scoreList = []
        for parent in child2parents[child]:
            scoreList.append(parent_child2score[parent + " -> " + child])

        child_parents = child + ": " + "|".join(list(child2parents[child]))
        child_parents2avg[child_parents] = sum(scoreList) / float(len(scoreList))

    sorted_child_parents2avg = sorted(child_parents2avg.items(), key = lambda e:e[1], reverse = True)

    output = open(args.genre + "sorted_child_parents2avg.txt", "w")
    for item in sorted_child_parents2avg:
        output.write(str(item[1]) + "\n")
        child = item[0].split(": ")[0]
        parents = item[0].split(": ")[1]

        output.write(child + " | " + str(item[1]) + "\n")
        for e in parents.split("|"):
            output.write(" <- " + e + " | " + str(parent_child2num[e + " -> " + child]) + "\n")
        output.write("\n\n")
    output.close()






