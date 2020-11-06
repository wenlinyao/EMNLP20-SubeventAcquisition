import ast, nltk, argparse, pickle, time
import networkx as nx


def remove_brackets(sentence):
    sentence = sentence.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
    return " ".join(sentence.split())


class Sentence:
    def __init__ (self, wordList, lemmaList, POSList, NERList, Timex, parseTree, dependencyList):
        """
        Use "wordList, lemmaList, POSList, NERList, Timex, parseTree, dependencyList" to initialize Sentence object.
        """
        self.wordList = wordList
        self.lemmaList = lemmaList
        self.POSList = POSList
        self.NERList = NERList
        self.Timex = Timex
        self.parseTree = parseTree
        self.dependencyList = dependencyList
        self.length = len(wordList)

class Event:
    def __init__ (self, indexList, wordList):
        self.indexList = ["<"] + indexList + [">"]
        self.wordList = ["<"] + wordList + [">"]
        for word in wordList:
            if word[0] == "[":
                self.trigger_word = word
        for index in indexList:
            if index[0] == "[":
                self.trigger_idx = index

class EventPair:
    def __init__ (self, string, freq):
        words = string.split()
        self.event1 = ""
        self.event2 = ""
        self.event1_trigger = ""
        self.event2_trigger = ""
        self.relation = ""
        self.freq = freq
        relation_idx = -1
        angle_brackets_count = 0
        event_triggerList = []
        end_idx = -1
        for i, word in enumerate(words):
            
            if word in ["<", ">"]:
                angle_brackets_count += 1
                if angle_brackets_count == 4:
                    end_idx = i
                    break
                continue

            if angle_brackets_count == 2:
                self.relation = word.replace("prep_", "nmod:")
                relation_idx = i
            if word[0] == "[":
                event_triggerList.append(word)
        
        self.event1 = " ".join(words[:relation_idx])
        self.event2 = " ".join(words[relation_idx+1:end_idx+1])

        self.event1_trigger = event_triggerList[0]
        self.event2_trigger = event_triggerList[1]
        

class CollapsedDependency:
    def __init__ (self, type, gov, dep):
        self.type = type
        self.gov = gov
        self.dep = dep

class CoreferenceMention:
    def __init__ (self, sentence_id, start, end, head):
        self.sentence_id = sentence_id
        self.start = start
        self.end = end
        self.head = head


def load_not_event(file1, file2, file3):
    input_file = open(file1)
    invalid_noun_set = set()
    for line in input_file:
        if not line.strip():
            continue
        if line[0] == "x":
            invalid_noun_set.add("["+line.split()[1]+"]")

    input_file.close()

    input_file = open(file2)
    for line in input_file:
        if not line.strip():
            continue
        if line[0] == "#":
            continue
        invalid_noun_set.add("["+line.split()[0]+"]")
    input_file.close()

    input_file = open(file3)
    for line in input_file:
        if not line.strip():
            continue
        if line[0] == "#":
            invalid_noun_set.add("["+line.split()[-1]+"]")
    input_file.close()

    return invalid_noun_set



def load_event_nouns(file):
    event_nouns = set()
    input_file = open(file, "r")
    count = 0
    for line in input_file:
        count += 1
        words = line.split()
        if words[0] == "#":
            continue
        event_nouns.add(words[-1])
    input_file.close()
    return event_nouns

def load_report_words(file):
    report_words = set()
    input_file = open(file, "r")
    for line in input_file:
        if not line.strip():
            continue
        report_words.add(line.split()[0])
    input_file.close()
    return report_words

def load_events_from_list(file):
    input_file = open(file, "r").readlines()
    assert len(input_file) == 1
    event_nouns = set(ast.literal_eval(input_file[0]))
    return event_nouns
# "[have]", "[believe]"
invalid_verb_set = set(["[belong]", "[engage]", "[contain]", "[involve]", "[relate]", "[go]",
                        "[include]", "[seem]", "[remain]", "[live]", "[stay]",
                        "[say]", "[think]", "[look]", "[feel]"])
invalid_noun_set = load_not_event("../dic/filtered_sort_by_sum_frequency.txt", "../dic/words_filter.txt", "../dic/nounWords5000.txt")
event_noun_set = set(load_event_nouns("../dic/all_noun_events.txt"))
light_verbs = set(["be", "do", "make", "get", "give", "take", "let", "set", "put", "have", "keep", "go", "work", "play", "stop", "run", "start", "become", "come", "hold"])
B_light_verbs = set(["[be]", "[do]", "[make]", "[get]", "[give]", "[take]", "[let]", "[set]", "[put]", "[have]", "[keep]", "[go]", "[work]", "[play]", "[stop]", "[run]", "[start]", "[become]", "[come]", "[hold]"])
pronouns = set(["a", "an", "the", "it", "this", "that", "what", "which", "these", "those", "any", "anything", "some", "something", "other", "misc", 
                "all", "each", "another", "either", "much", "many", "lot", "total", "more",
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])
person_pronouns = set(["i", "you", "he", "she", "we", "they", "who", "whom", "man", "woman",
                "myself", "yourself", "himself", "herself", "ourselves", "themselves",
                "anyone", "anybody", "someone", "somebody", "everybody"])

def extract_all_pairs(file):
    input_lines = open(file, "r")
    all_pairs = set()
    for line in input_lines:
        if not line.strip():
            continue
        fields = line.split(" | ")
        all_pairs.add(fields[0].replace("\n", ""))
    input_lines.close()
    return all_pairs

# Extract seed pairs from ranked list.
# Usage: valid_pairs = extract_valid_pairs("../run_extract_event_pair_nmod2/news/sorted_parent_child2num.txt", 2.0)
def extract_valid_pairs(file, score_threshold):
    valid_pairs = set()
    input_file = open(file, "r")
    
    for line in input_file:
        items = line.split(" | ")
        pair = items[0]

        score = float(items[1])
        freqList = ast.literal_eval(items[2])
        if (score > score_threshold and freqList[1] != freqList[-1]) or freqList[0] >= 2*score_threshold:
            valid_pairs.add(pair)

    input_file.close()
    return valid_pairs

def extract_seed_cand_pairs_NoClean(file, score_threshold):
    seed_pairs = set()
    candidate_pairs = set()
    input_file = open(file, "r")
    
    for line in input_file:
        items = line.split(" | ")
        eventpair = EventPair(items[0], -1)
        assert eventpair.relation == "->"
        pair = eventpair.event1 + " -> " + eventpair.event2
        R_pair = eventpair.event2 + " <- " + eventpair.event1
        score = float(items[1])
        freqList = ast.literal_eval(items[2])
        # remove all pairs that only co-occur with "in" from seed
        if (score > score_threshold and freqList[1] != freqList[-1]) or freqList[0] >= 2*score_threshold:
            seed_pairs.add(pair)
            seed_pairs.add(R_pair)
        elif score >= 1.1 or freqList[0] >= 1:
            candidate_pairs.add(pair)
            candidate_pairs.add(R_pair)

    input_file.close()
    return seed_pairs, candidate_pairs

def extract_seed_cand_pairs_Clean(seed_file, all_file):
    seed_pairs = set()
    candidate_pairs = set()

    input_lines = open(seed_file, "r")
    for line in input_lines:
        items = line.split(" | ")
        eventpair = EventPair(items[0], -1)
        assert eventpair.relation == "->"
        pair = eventpair.event1 + " -> " + eventpair.event2
        R_pair = eventpair.event2 + " <- " + eventpair.event1
        seed_pairs.add(pair)
        seed_pairs.add(R_pair)
    input_lines.close()

    input_lines = open(all_file, "r")

    for line in input_lines:
        items = line.split(" | ")
        score = float(items[1])
        freqList = ast.literal_eval(items[2])
        if score >= 1.1 or freqList[0] >= 1:
            eventpair = EventPair(items[0], -1)
            assert eventpair.relation == "->"
            pair = eventpair.event1 + " -> " + eventpair.event2
            R_pair = eventpair.event2 + " <- " + eventpair.event1
            if pair not in seed_pairs:
                candidate_pairs.add(pair)
                candidate_pairs.add(R_pair)

    input_lines.close()

    return seed_pairs, candidate_pairs

def extract_candidates_conj(file, threshold):
    input_lines = open(file, "r")
    candidate_pairs = set()
    for line in input_lines:
        fields = line.split(" | ")
        if float(fields[1]) >= threshold:
            candidate_pairs.add(fields[0])
    input_lines.close()
    return candidate_pairs

def extract_candidates_surrounding(file, threshold):
    input_lines = open(file, "r")
    candidate_pairs = set()
    for line in input_lines:
        fields = line.split(" | ")
        if float(fields[1].replace("\n", "")) >= threshold:
            candidate_pairs.add(fields[0])
    input_lines.close()
    return candidate_pairs

def extract_CT_pairs(file, threshold):
    input_lines = open(file, "r")
    C_pairs = set()
    T_pairs = set()
    for line in input_lines:
        fields = line.split(" | ")
        scores = ast.literal_eval(fields[1])
        if scores[0] >= threshold:
            C_pairs.add(fields[0].lower())
        if scores[1] >= threshold:
            T_pairs.add(fields[0].lower())
    input_lines.close()
    return C_pairs, T_pairs

def extract_BERT_predicted_pairs(file, threshold):
    input_lines = open(file, "r")
    pairs = set()
    for line in input_lines:
        fields = line.split(" | ")
        countList = ast.literal_eval(fields[1])
        if float(countList[0]+1)/float(sum(countList)+1) >= 0.5 and countList[0] >= threshold:
            pairs.add(fields[0])
    input_lines.close()
    return pairs

def extract_trigger_pair2score(valid_pairs_file, threshold):
    input_file = open(valid_pairs_file, "r")
    trigger_pair2score = {}
    for line in input_file:
        if not line.strip():
            continue
        items = line.split(" | ")
        if len(items) == 1:
            score = 1.0
        elif items[1][0] == "[":
            score = sum(ast.literal_eval(items[1]))
        else:
            score = float(items[1])
        
        if score < threshold:
            continue
        eventpair = EventPair(items[0].replace("\n",""), -1)
        if eventpair.event1_trigger in B_light_verbs or eventpair.event2_trigger in B_light_verbs:
            continue
        trigger_pair = eventpair.event1_trigger + " " + eventpair.event2_trigger
        if trigger_pair not in trigger_pair2score:
            trigger_pair2score[trigger_pair] = 0.0
        trigger_pair2score[trigger_pair] += score
    input_file.close()
    return trigger_pair2score

def extract_trigger_pair2score_new(valid_pairs):
    trigger_pair2score = {}
    for pair in valid_pairs:
        eventpair = EventPair(pair, -1)
        if eventpair.event1_trigger in B_light_verbs or eventpair.event2_trigger in B_light_verbs:
            continue
        trigger_pair = eventpair.event1_trigger + " " + eventpair.event2_trigger
        if trigger_pair not in trigger_pair2score:
            trigger_pair2score[trigger_pair] = 0.0
        trigger_pair2score[trigger_pair] += 1.0
    return trigger_pair2score

def extract_trigger_pair2scores(valid_pairs_file, threshold):
    input_file = open(valid_pairs_file, "r")
    trigger_pair2scores = {}
    for line in input_file:
        if not line.strip():
            continue
        items = line.split(" | ")
        
        scores = ast.literal_eval(items[1])
        if max(scores) < threshold:
            continue
        eventpair = EventPair(items[0], -1)
        trigger_pair = eventpair.event1_trigger + " " + eventpair.event2_trigger
        if trigger_pair not in trigger_pair2scores:
            trigger_pair2scores[trigger_pair] = [0.0, 0.0]
        trigger_pair2scores[trigger_pair][0] += scores[0]
        trigger_pair2scores[trigger_pair][1] += scores[1]
    input_file.close()
    return trigger_pair2scores

def CT_trigger_pair2score(pairs_file, threshold):
    input_file = open(pairs_file, "r")
    C_trigger_pair2score = {}
    T_trigger_pair2score = {}
    for line in input_file:
        if not line.strip():
            continue
        items = line.split(" | ")
        
        scores = ast.literal_eval(items[1])
        if scores[0] < threshold and scores[1] < threshold:
            continue
        eventpair = EventPair(items[0], -1)
        trigger_pair = eventpair.event1_trigger + " " + eventpair.event2_trigger

        if scores[0] >= threshold:
            if trigger_pair not in C_trigger_pair2score:
                C_trigger_pair2score[trigger_pair] = 0.0
            C_trigger_pair2score[trigger_pair] += scores[0]

        if scores[1] >= threshold:
            if trigger_pair not in T_trigger_pair2score:
                T_trigger_pair2score[trigger_pair] = 0.0
            T_trigger_pair2score[trigger_pair] += scores[1]
        
    input_file.close()
    return C_trigger_pair2score, T_trigger_pair2score


def translate_NER(sentence, idx):
    
    try:
        if sentence.lemmaList[idx].lower() in person_pronouns:
            return "PERSON"
        #elif sentence.wordList[idx][0].isupper() and sentence.NERList[idx] != 'O':
        elif sentence.NERList[idx] != 'O':
            return sentence.NERList[idx]
        else:
            #return sentence.wordList[idx]
            return sentence.lemmaList[idx]
    except:
        print("ERROR word idx:", idx)
        print(sentence.length)
        print(" ".join(sentence.wordList))
        print("|".join(sentence.dependencyList))
        return("")

def detect_event_trigger(sentence, idx):
    if sentence.lemmaList[idx] in ["be", "seem"]:
        return False
    if sentence.POSList[idx][0] == "V":
        if "[" + sentence.lemmaList[idx] + "]" in invalid_verb_set:
            return False
        else:
            return True
    elif sentence.POSList[idx][0] == "N":
        if "[" + sentence.lemmaList[idx] + "]" in invalid_noun_set:
            return False
        elif sentence.lemmaList[idx] in event_noun_set:
            return True
    else:
        return False
        

def document2sentenceList(document):
    """
    Parse document to sentenceList
    """
    sentenceList = []
    # use sentence id as boundary
    paragraph_boundary = []
    coreferenceList = []
    for line in document:
        words = line.split()
        if words[0] == "<paragraph>":
            paragraphs = line.split("|")
            for paragraph in paragraphs[1:]:
                sentence_ids = paragraph.split()
                if len(sentence_ids) == 0:
                    continue
                boundary = int(sentence_ids[-1])
                paragraph_boundary.append(boundary)
        elif words[0] == "<word>":
            wordList = words
        elif words[0] == "<lemma>":
            lemmaList = words
        elif words[0] == "<POS>":
            POSList = words
        elif words[0] == "<NER>":
            NERList = words
        elif words[0] == "<Timex>":
            Timex = ast.literal_eval(line.replace("<Timex> ", "").replace("\n", ""))
        elif words[0] == "<parse>":
            parseTree = " ".join(words[1:])
        elif words[0] == "<collapsed-ccprocessed-dependencies>":
            dependencyList = line.split("|")
            sentence = Sentence(wordList, lemmaList, POSList, NERList, Timex, parseTree, dependencyList[1:-1])
            sentenceList.append(sentence)
        elif words[0] == "<coreference>":
            coreferenceList = ast.literal_eval(line.replace("<coreference> ", "").replace("\n", ""))
    return sentenceList, paragraph_boundary, coreferenceList


def old_extract_event_with_arg(sentence, word_id):
    """
    Extract event arguments by scanning dependency tree
    :param word_id: event trigger index
    """
    collapsed_dependenciesList = []
    for d in sentence.dependencyList:
        words = d.split()
        if len(words) != 3:
            continue
        collapsed_dependenciesList.append(CollapsedDependency(words[1], words[0], words[2]))
    
    index_result = []
    words_result = []
    temp_index_result = []
    temp_words_result = []
    noun_flag = 0
    obj_flag = 0
    words_result.append( '[' + sentence.lemmaList[word_id] + ']')
    index_result.append('[' + str(word_id) + ']')
    
    type2dep = {} # map dependency type to its dependent word
    type2gov = {} # map dependency type to its governor word


    for element in collapsed_dependenciesList:
        if int(element.gov) == int(word_id):
            type2dep[element.type] = element.dep
        if int(element.dep) == int(word_id):
            type2gov[element.type] = element.gov

    if 'neg' in type2dep:
        dep = type2dep['neg']
        words_result.insert(0, 'not')
        index_result.insert(0, 'not')

    if sentence.POSList[word_id][0] =='N':
        if 'prep_of' in type2dep:
            dep = type2dep['prep_of']
            translated = translate_NER (sentence, int(dep))
            words_result.append('of')
            index_result.append('of')
            words_result.append(translated)
            index_result.append(str(dep))
        elif 'prep_by' in type2dep:
            dep = type2dep['prep_by']
            translated = translate_NER (sentence, int(dep))
            words_result.append('by')
            index_result.append('by')
            words_result.append(translated)
            index_result.append(str(dep))
        else:
            prep_other_flag = 0
            prep_other_type = ""
            dep = 0
            for ele in type2dep:
                if 'prep_' in ele:
                    prep_other_flag = 1
                    prep_other_type = ele.replace("prep_", "")
                    dep = type2dep[ele]
            if prep_other_flag == 1 and prep_other_type not in ["during", "in", "amid", "within", "throughout", "including"]:
                translated = translate_NER (sentence, int(dep))
                words_result.append(prep_other_type)
                index_result.append(prep_other_type)
                words_result.append(translated)
                index_result.append(str(dep))
        
    elif sentence.POSList[word_id][0] =='V':
        verb_alone = True
        if 'prt' in type2dep:
            dep = type2dep['prt']
            words_result.append(sentence.wordList[int(dep)])
            index_result.append(str(dep))
        if 'dobj' in type2dep:
            dep = type2dep['dobj']
            translated = translate_NER (sentence, int(dep))
            words_result.append(translated)
            index_result.append(str(dep))
            verb_alone = False
        elif 'nsubjpass' in type2dep:
            dep = type2dep['nsubjpass']

            words_result = []
            words_result.append( '[' + sentence.wordList[word_id] + ']') # use word instead of lemma

            translated = translate_NER (sentence, int(dep))
            words_result.insert(0, 'be') # this order is for the convenience of words_result.insert()
            index_result.insert(0, 'be')
            words_result.insert(0, translated)
            index_result.insert(0, str(dep))
            verb_alone = False
        elif 'xcomp' in type2dep:
            dep = type2dep['xcomp']
            translated = translate_NER (sentence, int(dep))
            if sentence.wordList[int(dep) - 1] == 'to':
                words_result.append('to')
                index_result.append(str(int(dep) - 1))
                words_result.append(translated)
                index_result.append(str(dep))
            else:
                words_result.append(translated)
                index_result.append(str(dep))
            verb_alone = False
        elif 'nsubj' in type2dep:
            dep = type2dep['nsubj']
            translated = translate_NER (sentence, int(dep))
            words_result.insert(0, translated)
            index_result.insert(0, str(dep))
            verb_alone = False
        elif 'amod' in type2gov:
            gov = type2gov['amod']
            translated = translate_NER (sentence, int(gov))
            words_result.append(translated)
            index_result.append(str(gov))
            verb_alone = False
        else:
            prep_other_flag = 0
            prep_other_type = ""
            dep = 0
            for ele in type2dep:
                if 'prep_' in ele:
                    prep_other_flag = 1
                    prep_other_type = ele.replace("prep_", "")
                    dep = type2dep[ele]
            if prep_other_flag == 1 and prep_other_type not in ["during", "in", "amid", "within", "throughout", "including"]:
                translated = translate_NER (sentence, int(dep))
                words_result.append(prep_other_type)
                index_result.append(prep_other_type)
                words_result.append(translated)
                index_result.append(str(dep))
                verb_alone = False

    if sentence.POSList[word_id][0] == 'V' and sentence.POSList[word_id] != 'VBG' and verb_alone == True:
        return index_result, words_result, "verb_alone"
    return index_result, words_result, ""


def new_extract_event_with_arg(sentence, word_id):
    """
    Extract event arguments by scanning dependency tree
    :param word_id: event trigger index
    """
    collapsed_dependenciesList = []
    for d in sentence.dependencyList:
        words = d.split()
        if len(words) != 3:
            continue
        collapsed_dependenciesList.append(CollapsedDependency(words[1], words[0], words[2]))
    
    index_result = []
    words_result = []
    temp_index_result = []
    temp_words_result = []
    noun_flag = 0
    obj_flag = 0
    words_result.append( '[' + sentence.lemmaList[word_id] + ']')
    index_result.append('[' + str(word_id) + ']')
    
    type2dep = {} # map dependency type to its dependent word
    type2gov = {} # map dependency type to its governor word


    for element in collapsed_dependenciesList:
        if int(element.gov) == int(word_id):
            type2dep[element.type] = element.dep
        if int(element.dep) == int(word_id):
            type2gov[element.type] = element.gov

    if 'neg' in type2dep:
        dep = type2dep['neg']
        words_result.insert(0, 'not')
        index_result.insert(0, 'not')

    if sentence.POSList[word_id][0] =='N':
        if 'nmod:of' in type2dep:
            dep = type2dep['nmod:of']
            translated = translate_NER (sentence, int(dep))
            words_result.append('of')
            index_result.append('of')
            words_result.append(translated)
            index_result.append(str(dep))
        elif 'nmod:by' in type2dep:
            dep = type2dep['nmod:by']
            translated = translate_NER (sentence, int(dep))
            words_result.append('by')
            index_result.append('by')
            words_result.append(translated)
            index_result.append(str(dep))
        else:
            prep_other_flag = 0
            prep_other_type = ""
            dep = 0
            for ele in type2dep:
                if 'nmod:' in ele and ele != "nmod:poss":
                    prep_other_flag = 1
                    prep_other_type = ele.replace("nmod:", "")
                    dep = type2dep[ele]
            if prep_other_flag == 1 and prep_other_type not in ["during", "in", "amid", "within", "throughout", "including"]:
                translated = translate_NER (sentence, int(dep))
                words_result.append(prep_other_type)
                index_result.append(prep_other_type)
                words_result.append(translated)
                index_result.append(str(dep))
        
    elif sentence.POSList[word_id][0] =='V':
        verb_alone = True
        if 'compound:prt' in type2dep:
            dep = type2dep['compound:prt']
            words_result.append(sentence.wordList[int(dep)])
            index_result.append(str(dep))
        if 'dobj' in type2dep:
            dep = type2dep['dobj']
            translated = translate_NER (sentence, int(dep))
            words_result.append(translated)
            index_result.append(str(dep))
            verb_alone = False
        elif 'nsubjpass' in type2dep:
            dep = type2dep['nsubjpass']

            words_result = []
            words_result.append( '[' + sentence.wordList[word_id] + ']') # use word instead of lemma

            translated = translate_NER (sentence, int(dep))
            words_result.insert(0, 'be') # this order is for the convenience of words_result.insert()
            index_result.insert(0, 'be')
            words_result.insert(0, translated)
            index_result.insert(0, str(dep))
            verb_alone = False
        elif 'xcomp' in type2dep:
            dep = type2dep['xcomp']
            translated = translate_NER (sentence, int(dep))
            if sentence.wordList[int(dep) - 1] == 'to':
                words_result.append('to')
                index_result.append(str(int(dep) - 1))
                words_result.append(translated)
                index_result.append(str(dep))
            else:
                words_result.append(translated)
                index_result.append(str(dep))
            verb_alone = False
        elif 'nsubj' in type2dep:
            dep = type2dep['nsubj']
            translated = translate_NER (sentence, int(dep))
            words_result.insert(0, translated)
            index_result.insert(0, str(dep))
            verb_alone = False
        elif 'amod' in type2gov:
            gov = type2gov['amod']
            translated = translate_NER (sentence, int(gov))
            words_result.append(translated)
            index_result.append(str(gov))
            verb_alone = False
        else:
            prep_other_flag = 0
            prep_other_type = ""
            dep = 0
            for ele in type2dep:
                if 'nmod:' in ele and ele != "nmod:poss":
                    prep_other_flag = 1
                    prep_other_type = ele.replace("nmod:", "")
                    dep = type2dep[ele]
            if prep_other_flag == 1 and prep_other_type not in ["during", "in", "amid", "within", "throughout", "including"]:
                translated = translate_NER (sentence, int(dep))
                words_result.append(prep_other_type)
                index_result.append(prep_other_type)
                words_result.append(translated)
                index_result.append(str(dep))
                verb_alone = False

    # filter verb event trigger with no arguments
    if sentence.POSList[word_id][0] == 'V' and sentence.POSList[word_id] != 'VBG' and verb_alone == True:
        return index_result, words_result, "verb_alone"

    return index_result, words_result, ""

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

#linking_verbs = set(["[include]", "[seem]", "[remain]"])
def validate_eventpair(eventpair):
    invalid_triggers = set(["[attempt]", "[location]", "[colony]"])
    if eventpair.event1_trigger in invalid_triggers or eventpair.event2_trigger in invalid_triggers:
        return False
    if eventpair.event1_trigger in ["[lung]", "[pain]", "[agony]", "[misery]", "[astonishment]", "[agitation]", "[sympathy]"]:
        return False
    if eventpair.event1_trigger == eventpair.event2_trigger:
        return False
    if eventpair.event1 == "< [take] place >" or eventpair.event2 == "< [take] place >":
        return False
    if eventpair.event1 == "< [take] part >" or eventpair.event2 == "< [take] part >":
        return False
    if eventpair.event1 == "< [play] role >" or eventpair.event2 == "< [play] role >":
        return False
    if " [have] to " in eventpair.event1 or " [have] to " in eventpair.event2:
        return False
    
    event1_wordList = eventpair.event1.split()
    event2_wordList = eventpair.event2.split()

    if len(event1_wordList) == 3 and len(event2_wordList) == 3:
        return False

    if hasNumbers(eventpair.event1) or hasNumbers(eventpair.event2):
        return False

    if " "+remove_brackets(eventpair.event1_trigger)+" " in eventpair.event2:
        return False
    if " "+remove_brackets(eventpair.event2_trigger)+" " in eventpair.event1:
        return False
    #if eventpair.event1_trigger in invalid_verb_set or eventpair.event2_trigger in invalid_verb_set:
    #    return False

    eventpair_str = eventpair.event1 + eventpair.event2
    stock_words = ["percent", "number", "money", "point"]

    for stock_word1 in stock_words:
        for stock_word2 in stock_words:
            if stock_word1 in eventpair_str and stock_word2 in eventpair_str:
                return False

    stock_nonevents = ["[share]", "[index]", "[bond]"]
    # < [share] > -> < [decline] number >
    for stock_nonevent in stock_nonevents:
        if stock_nonevent in eventpair_str:
            return False
    
    # light verb trigger without arguments
    if eventpair.event1_trigger.replace("[", "").replace("]", "") in light_verbs:
        if len(event1_wordList) == 3:
            return False
        # Whether the light verb has no object
        elif event1_wordList[-1] == ">" and event1_wordList[-2][-1] == "]":
            return False

    if eventpair.event2_trigger.replace("[", "").replace("]", "") in light_verbs:
        if len(event2_wordList) == 3:
            return False
        # Whether the light verb has no object
        elif event2_wordList[-1] == ">" and event2_wordList[-2][-1] == "]":
            return False

    if len((set(event1_wordList) | set(event2_wordList)) & pronouns) != 0:
        return False
    return True

def clean_eventpair(eventpair_str):
    new_eventpair_str = []
    for word in eventpair_str.split():
        if word in person_pronouns:
            new_eventpair_str.append("person")
        else:
            new_eventpair_str.append(word)
    return " ".join(new_eventpair_str)
    

def get_path2(sentence, index1, index2):
    collapsed_dependenciesList = []
    for d in sentence.dependencyList:
        words = d.split()
        if len(words) != 3:
            continue
        collapsed_dependenciesList.append(CollapsedDependency(words[1], words[0], words[2]))
    for entity in collapsed_dependenciesList:
        head = entity.gov
        tail = entity.dep
        if (head == index1 and tail == index2) or (head == index2 and tail == index1):
            return entity.type
    return None

def get_path(sentence, index1, index2):
    collapsed_dependenciesList = []
    gov_dep2type = {}
    dep_gov2type = {}
    dep2gov = {}
    for d in sentence.dependencyList:
        words = d.split()
        if len(words) != 3:
            continue
        collapsed_dependenciesList.append(CollapsedDependency(words[1], words[0], words[2])) # (type, gov, dep)
        gov_dep2type[words[0] + " " + words[2]] = words[1]
        dep_gov2type[words[2] + " " + words[0]] = "R-" + words[1]
        dep2gov[words[2]] = words[0]

    valid_index = []
    event1_index = index1
    event2_index = index2

    G = nx.Graph()
    pobj_dict = {}
    for entity in collapsed_dependenciesList:
        head = entity.gov
        tail = entity.dep
        #if (head == event1_index and tail == event2_index) or (tail == event1_index and head == event2_index):
        #    return None, None
        edge = (head, tail)
        G.add_edge(*edge)

    try: 
        path = nx.shortest_path(G, source = event1_index, target = event2_index)
    except:
        return None, None, None
    
    #valid_index = list(set(path) - set([event1_index, event2_index]))
    valid_index = list(set(path))

    word_path = []
    for i in range(0, len(path)-1):
        if i != 0:
            word_path.append(sentence.lemmaList[int(path[i])]) # ignore source and target node
        if path[i] + " " + path[i+1] in gov_dep2type:
            word_path.append(gov_dep2type[path[i] + " " + path[i+1]])
        elif path[i] + " " + path[i+1] in dep_gov2type:
            word_path.append(dep_gov2type[path[i] + " " + path[i+1]])

    if len(word_path) == 1 and event1_index in dep2gov:
        extra_w_idx = dep2gov[event1_index]
        if int(extra_w_idx) < int(event1_index) and int(extra_w_idx) != 0 and sentence.POSList[int(extra_w_idx)][0] != 'N':
            extra_w = sentence.lemmaList[int(extra_w_idx)]
            extra_type = gov_dep2type[extra_w_idx + " " + event1_index]
            word_path.insert(0, "*")
            word_path.insert(0, extra_type)
            word_path.insert(0, extra_w)


    
    valid_index = map(int, valid_index) # change string list to int list
    valid_index.sort()

    path_pos = []
    for idx in valid_index:
        if idx == int(event1_index):
            path_pos.append("E1")
        elif idx == int(event2_index):
            path_pos.append("E2")
        else:
            path_pos.append(sentence.lemmaList[idx])

    path_pos = " ".join(path_pos)
    
    return valid_index, " ".join(word_path), path_pos

def detect_explicit_connector(sentence):
    connectorList = ["meanwhile ,", "meantime ,", "concurrently ,", "simultaneously ,"]
    for connector in connectorList:
        if connector in " ".join(sentence.wordList).lower():
            return True
    return False




