# coding: utf-8
import sys, codecs, pdb, re, math, pickle, random
sys.path.append("../../preprocess_RED/model/")
reload(sys)
sys.setdefaultencoding('utf-8')
import glob, xmltodict, time
import networkx as nx
from load_utilities import document2sentenceList
from model_utilities import get_context, extract_event_with_arg

def make_annotation(super_event, sub_event, sentenceList, file_name, Type):
    args_in_same_sent = super_event["s_idx"] == sub_event["s_idx"]

    if args_in_same_sent == True:
        context = get_context(sentenceList[super_event["s_idx"]], "tree_path_with_events", str(super_event["w_idx"]), str(sub_event["w_idx"]))
        
        index_result, words_result = extract_event_with_arg(sentenceList[super_event["s_idx"]], super_event["w_idx"])
        event1 = " ".join(words_result)
        for index in index_result:
            if index[0] == "[":
                mask1_idx = int(index.replace("[","").replace("]",""))
        
        index_result, words_result = extract_event_with_arg(sentenceList[sub_event["s_idx"]], sub_event["w_idx"])
        event2 = " ".join(words_result)
        for index in index_result:
            if index[0] == "[":
                mask2_idx = int(index.replace("[","").replace("]",""))

        sentence = sentenceList[super_event["s_idx"]]
        masked_sentence = [sentence.wordList[k] if k not in [mask1_idx, mask2_idx] else "[MASK]" for k in range(1, len(sentence.wordList))]

        annotation = {
            "filename": file_name,
            "flag": "within_sentence",
            "sentence1": sentenceList[super_event["s_idx"]],
            "sentence2": sentenceList[super_event["s_idx"]],
            "masked_sentence1":masked_sentence, 
            "masked_sentence2":masked_sentence,
            "event1_trigger": sentenceList[super_event["s_idx"]].wordList[super_event["w_idx"]],
            "event2_trigger": sentenceList[sub_event["s_idx"]].wordList[sub_event["w_idx"]],
            "event1_word_idx": super_event["w_idx"],
            "event2_word_idx": sub_event["w_idx"],
            "event1": event1,
            "event2": event2,
            "relation": Type,
            "context": context
        }

        if super_event["w_idx"] <= sub_event["w_idx"]:
            annotation["order_flag"] = "e1->e2"
        else:
            annotation["order_flag"] = "e2->e1"

        HiEve_annotationList.append(annotation)
    else:

        index_result, words_result = extract_event_with_arg(sentenceList[super_event["s_idx"]], super_event["w_idx"])
        event1 = " ".join(words_result)
        for index in index_result:
            if index[0] == "[":
                mask1_idx = int(index.replace("[","").replace("]",""))
        sentence = sentenceList[super_event["s_idx"]]
        masked_sentence1 = [sentence.wordList[k] if k not in [mask1_idx] else "[MASK]" for k in range(1, len(sentence.wordList))]

        
        index_result, words_result = extract_event_with_arg(sentenceList[sub_event["s_idx"]], sub_event["w_idx"])
        event2 = " ".join(words_result)
        for index in index_result:
            if index[0] == "[":
                mask2_idx = int(index.replace("[","").replace("]",""))

        sentence = sentenceList[sub_event["s_idx"]]
        masked_sentence2 = [sentence.wordList[k] if k not in [mask2_idx] else "[MASK]" for k in range(1, len(sentence.wordList))]

        annotation = {
            "filename": file_name,
            "flag": "cross_sentence",
            "sentence1": sentenceList[super_event["s_idx"]],
            "sentence2": sentenceList[sub_event["s_idx"]],
            "masked_sentence1":masked_sentence1, 
            "masked_sentence2":masked_sentence2,
            "event1_trigger": sentenceList[super_event["s_idx"]].wordList[super_event["w_idx"]],
            "event2_trigger": sentenceList[sub_event["s_idx"]].wordList[sub_event["w_idx"]],
            "event1_word_idx": super_event["w_idx"],
            "event2_word_idx": sub_event["w_idx"],
            "event1": event1,
            "event2": event2,
            "relation": Type,
            "context": None
        }

        if super_event["s_idx"] <= sub_event["s_idx"]:
            annotation["order_flag"] = "e1->e2"
        else:
            annotation["order_flag"] = "e2->e1"
    return annotation

def find_trigger(event_mention, text, sentenceList):

    max_window_size = 12
    shift = 0
    
    trigger_start = event_mention["Position"]
    for i in range(event_mention["Position"], len(text)):
        if text[i].isspace() or text[i] in ["\xe2\x80\x9d", "\""]:
            trigger_end = i
            break
    trigger_end = i
    fulltext = text.lower()
    #print(fulltext[trigger_start:trigger_end], event_mention.AnchorText.lower())
    if not event_mention["AnchorText"].lower() in fulltext[trigger_start:trigger_end]:
        print("Error:")
        print(fulltext[trigger_start:trigger_end], event_mention["AnchorText"].lower())
        #print(fulltext)
    window_chars = fulltext[trigger_start:trigger_end]
    for i in range(1, max_window_size):
        if trigger_start - i < 0:
            break
        if fulltext[trigger_start-i] in [" ", "\t", "\n", "\r"]:
            continue
        if fulltext[trigger_start-i].isalpha() == False and fulltext[trigger_start-i].isdigit() == False:
            break
        shift += 1
        window_chars = fulltext[trigger_start - i] + window_chars

    for i in range(0, max_window_size - 1):
        if trigger_end + i >= len(fulltext):
            break
        if fulltext[trigger_end + i] in [" ", "\t", "\n", "\r"]:
            continue
        if fulltext[trigger_end + i].isalpha() == False and fulltext[trigger_end + i].isdigit() == False:
            break
        window_chars = window_chars + fulltext[trigger_end + i]
    

    doc_str = ""
    charId2SentenceIdWordId = {}
    char_count = -1
    for s_idx, sentence in enumerate(sentenceList):
        for w_idx, word in enumerate(sentence.wordList):
            if w_idx == 0:
                continue
            for char in word:
                doc_str += char.lower()
                char_count += 1
                charId2SentenceIdWordId[char_count] = [s_idx, w_idx]
    #print(doc_str)

    try:
        result_idx = doc_str.index(window_chars)
        [s_idx, w_idx] = charId2SentenceIdWordId[result_idx + shift]
        """
        print "***********", window_chars, " ".join(sentenceList[s_idx].wordList)
        print fulltext[trigger_start:trigger_end], sentenceList[s_idx].wordList[w_idx]
        raw_input("continue?")
        """
        return s_idx, w_idx

    except ValueError:
        
        print "###########", window_chars, fulltext[trigger_start:trigger_end]
        print "$$$$$$$$$$$", doc_str
        raw_input("continue?")
        
        return None, None

def transitive_closure(hierarchy_relationList):
    cluster_idx = -1
    cluster_idx2EventIDs = {}
    for relation in hierarchy_relationList:
        SuperEventID = relation["SuperEventID"]
        SubEventID = relation["SubEventID"]
        if relation["Type"] == "Coref":
            found_flag = False
            for cluster_idx in cluster_idx2EventIDs:
                if SuperEventID in cluster_idx2EventIDs[cluster_idx] or SubEventID in cluster_idx2EventIDs[cluster_idx]:
                    found_flag = True
                    cluster_idx2EventIDs[cluster_idx].add(SuperEventID)
                    cluster_idx2EventIDs[cluster_idx].add(SubEventID)
            if found_flag == False:
                cluster_idx += 1
                cluster_idx2EventIDs[cluster_idx] = set([SuperEventID, SubEventID])
    for relation in hierarchy_relationList:
        SuperEventID = relation["SuperEventID"]
        SubEventID = relation["SubEventID"]
        if relation["Type"] == "SuperSub":
            found_flag = False
            for cluster_idx in cluster_idx2EventIDs:
                if SuperEventID in cluster_idx2EventIDs[cluster_idx]:
                    found_flag = True
                    
            if found_flag == False:
                cluster_idx += 1
                cluster_idx2EventIDs[cluster_idx] = set([SuperEventID])

            found_flag = False
            for cluster_idx in cluster_idx2EventIDs:
                if SubEventID in cluster_idx2EventIDs[cluster_idx]:
                    found_flag = True
                    
            if found_flag == False:
                cluster_idx += 1
                cluster_idx2EventIDs[cluster_idx] = set([SubEventID])

    EventID2cluster_idx = {}
    for cluster_idx in cluster_idx2EventIDs:
        for EventID in cluster_idx2EventIDs[cluster_idx]:
            EventID2cluster_idx[EventID] = cluster_idx

    G = nx.DiGraph()
    for relation in hierarchy_relationList:
        SuperEventID = relation["SuperEventID"]
        SubEventID = relation["SubEventID"]
        if relation["Type"] == "SuperSub":
            head = EventID2cluster_idx[SuperEventID]
            tail = EventID2cluster_idx[SubEventID]
            edge = (head, tail)
            G.add_edge(*edge)

    cluster_idxList = cluster_idx2EventIDs.keys()

    new_hierarchy_relationList = []
    for cluster_idx1 in cluster_idxList:
        for cluster_idx2 in cluster_idxList:
            if cluster_idx1 == cluster_idx2:
                continue
            try:
                path = nx.shortest_path(G, source=cluster_idx1, target=cluster_idx2)
                for SuperEventID in cluster_idx2EventIDs[cluster_idx1]:
                    for SubEventID in cluster_idx2EventIDs[cluster_idx2]:
                        new_hierarchy_relationList.append({"SuperEventID": SuperEventID, "SubEventID": SubEventID, "Type": "SuperSub"})
            except:
                pass

    for cluster_idx in cluster_idx2EventIDs:
        EventIDs = list(cluster_idx2EventIDs[cluster_idx])
        length = len(EventIDs)
        if length == 1:
            continue
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                new_hierarchy_relationList.append({"SuperEventID": EventIDs[i], "SubEventID": EventIDs[j], "Type": "Coref"})

    return new_hierarchy_relationList

if __name__ == "__main__":
    random.seed(11)
    folder = "../hievents/"
    HiEve_annotationList = []

    noun_events = set()

    SuperSub_count = 0
    Coref_count = 0
    other_relation_count = 0
    for file in glob.glob(folder + "*.xml"):
        print(file)
        fd = open(file)
        doc = xmltodict.parse(fd.read())
        
        text = doc["ArticleInfo"]["Text"]
        file_name = file.split("/")[-1]
        if file_name in ["article-13218.xml", "article-15897.xml"]:
            continue
        sentenceList, paragraph_boundary, coreferenceList = document2sentenceList(open("../new_run/"+"new_"+file_name.replace(".xml", ".txt.xml")).readlines())


        event_mentionList = []
        for event in doc["ArticleInfo"]["Events"]["EventMentionInfo"]:
            ID = event["ID"]
            AnchorText = event["AnchorText"]
            Type = event["Type"]
            Position = int(event["Position"])

            # find true position
            all_positions = [m.start() for m in re.finditer(AnchorText, text)]
            best_match = all_positions[0]
            min_distance = abs(Position-best_match)
            for p in all_positions:
                if abs(Position-p) < min_distance:
                    min_distance = abs(Position-p)
                    best_match = p

            event_mention = {"ID":ID, "AnchorText":AnchorText, "Type":Type, "Position":best_match}
            
            s_idx, w_idx = find_trigger(event_mention, text, sentenceList)

            if sentenceList[s_idx].POSList[w_idx][0] == "N":
                noun_events.add(sentenceList[s_idx].lemmaList[w_idx].lower())
            event_mention["s_idx"] = s_idx
            event_mention["w_idx"] = w_idx
            event_mentionList.append(event_mention)
            

        #pdb.set_trace()

        
        hierarchy_relationList = []

        annotated_pair_dict = set()


        for relation in doc["ArticleInfo"]["Relations"]["HierarchyRelationInfo"]:
            SuperEventID = relation["SuperEventID"]
            SubEventID = relation["SubEventID"]
            Type = relation["Type"]
            hierarchy_relation = {"SuperEventID":SuperEventID, "SubEventID":SubEventID, "Type":Type}
            hierarchy_relationList.append(hierarchy_relation)

        new_hierarchy_relationList = transitive_closure(hierarchy_relationList)
        #new_hierarchy_relationList = hierarchy_relationList

        for relation in new_hierarchy_relationList:
            SuperEventID = relation["SuperEventID"]
            SubEventID = relation["SubEventID"]
            Type = relation["Type"]

            if Type == "SuperSub":
                SuperSub_count += 1
            elif Type == "Coref":
                Coref_count += 1
            
            super_event, sub_event = None, None
            for event_mention in event_mentionList:
                if event_mention["ID"] == SuperEventID:
                    super_event = event_mention
                elif event_mention["ID"] == SubEventID:
                    sub_event = event_mention

            if super_event == None or sub_event == None or super_event["s_idx"] == None or sub_event["s_idx"] == None:
                #print(super_event)
                #print(sub_event)
                print("Relation: event mention not found!")
                continue

            dict_indicator = [super_event["s_idx"], super_event["w_idx"], sub_event["s_idx"], sub_event["w_idx"]]
            annotated_pair_dict.add("_".join([str(d) for d in dict_indicator]))
            dict_indicator = [sub_event["s_idx"], sub_event["w_idx"], super_event["s_idx"], super_event["w_idx"]]
            annotated_pair_dict.add("_".join([str(d) for d in dict_indicator]))

            annotation = make_annotation(super_event, sub_event, sentenceList, file_name, Type)
            HiEve_annotationList.append(annotation)

        for i in range(0, len(event_mentionList)):
            for j in range(i+1, len(event_mentionList)):
                event_mention1 = event_mentionList[i]
                event_mention2 = event_mentionList[j]
                #if event_mention1["s_idx"] != event_mention2["s_idx"]:
                #    continue
                dict_indicator = [event_mention1["s_idx"], event_mention1["w_idx"], event_mention2["s_idx"], event_mention2["w_idx"]]
                indicator = "_".join([str(d) for d in dict_indicator])
                dict_indicator = [event_mention2["s_idx"], event_mention2["w_idx"], event_mention1["s_idx"], event_mention1["w_idx"]]
                R_indicator = "_".join([str(d) for d in dict_indicator])
                if indicator in annotated_pair_dict or R_indicator in annotated_pair_dict:
                    continue
                other_relation_count += 1
                annotation = make_annotation(event_mention1, event_mention2, sentenceList, file_name, "Other")
                HiEve_annotationList.append(annotation)

    print("other_relation_count:", other_relation_count)
    print("SuperSub_count:", SuperSub_count)
    print("Coref_count:", Coref_count)

    pickle.dump(noun_events, open("HiEve_noun_events.p", "wb"))
    

    new_HiEve_annotationList = []
    Other_annotationList = []
    for annotation in HiEve_annotationList:
        if annotation["relation"] == "Coref":
            continue
        elif annotation["relation"] == "SuperSub":
            new_HiEve_annotationList.append(annotation)
        elif annotation["relation"] == "Other":
            Other_annotationList.append(annotation)

    random.shuffle(Other_annotationList)
    #new_HiEve_annotationList += Other_annotationList[:int(42094.0/3648.0*float(SuperSub_count))]

    output = open("HiEve_allRelations.txt", "w")
    new_other_relation_count = 0
    for annotation in new_HiEve_annotationList:

        if annotation["relation"] == "Other":
            new_other_relation_count += 1

        output.write("<START>\n")
        output.write("<filename> " + annotation["filename"] + "\n")
        output.write("<relation> " + annotation["relation"] + "\n")
        output.write("<event1_trigger> " + annotation["event1_trigger"] + "\n")
        output.write("<event2_trigger> " + annotation["event2_trigger"] + "\n")
        output.write("<order_flag> " + annotation["order_flag"] + "\n")
        output.write("<event1> " + annotation["event1"] + "\n")
        output.write("<event2> " + annotation["event2"] + "\n")
        output.write("<sentence1> " + " ".join(annotation["sentence1"].wordList[1:]) + "\n")
        output.write("<sentence2> " + " ".join(annotation["sentence2"].wordList[1:]) + "\n")
        output.write("<masked_sentence1> " + " ".join(annotation["masked_sentence1"]) + "\n") # <word> has already been removed.
        output.write("<masked_sentence2> " + " ".join(annotation["masked_sentence2"]) + "\n")
        output.write("<END>\n")
        output.write("\n\n")

    output.close()
    print("new_other_relation_count:", new_other_relation_count)


        





