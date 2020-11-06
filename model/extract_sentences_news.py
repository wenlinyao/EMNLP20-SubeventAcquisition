import sys
sys.path.append("../utilities/")
reload(sys)
sys.setdefaultencoding('utf-8')
import ast, glob, pickle, argparse
from multiprocessing import Process
from utilities import document2sentenceList, detect_event_trigger, old_extract_event_with_arg, load_previous_event_nouns



def extract_sentences_nmod(sentenceList, doc_id):
    # "prep_within"
    patternList = ["prep_during", "prep_in", "prep_amid", "prep_throughout", "prep_including", "prep_within"]

    result_sentences = []
    for sentence in sentenceList:
        sentence_str = " ".join(sentence.wordList)
        #if "``" in sentence_str:
        #    continue
        gov_set = set()
        for dependency in sentence.dependencyList:
            words = dependency.split()
            gov = int(words[0])
            rel = words[1]
            dep = int(words[2])
            for pattern in patternList:

                if pattern == rel:
                    connector = pattern

                    if detect_event_trigger(sentence, gov) == True and detect_event_trigger(sentence, dep) == True:
                        
                        index_result, words_result, tag1 = old_extract_event_with_arg(sentence, gov)
                        event1 = "< " + " ".join(words_result).lower() + " >"

                        index_result, words_result, tag2 = old_extract_event_with_arg(sentence, dep)
                        event2 = "< " + " ".join(words_result).lower() + " >"

                        if tag1 == "verb_alone" or tag2 == "verb_alone":
                            continue
                        result_sentences.append({"doc_id": doc_id, "sentence": sentence, "event_pair": event1 + " " + connector + " " + event2})

    return result_sentences


def extract_sentences_advcl(sentenceList, doc_id):

    result_sentences = []
    for sentence in sentenceList:
        sentence_str = " ".join(sentence.wordList)
        #if "``" in sentence_str:
        #    continue
        gov_set = set()
        for dependency in sentence.dependencyList:
            if "advcl" in dependency:
                l = dependency.split()
                gov = int(l[0])
                dep = int(l[2])

                window_content = " ".join(sentence.wordList[gov:dep]).lower()
                connector = None

                if "while" in window_content:
                    connector = "advcl:while"
                elif "when" in window_content:
                    connector = "advcl:when"

                if connector != None and detect_event_trigger(sentence, gov) == True and detect_event_trigger(sentence, dep) == True:
                    index_result, words_result, tag1 = old_extract_event_with_arg(sentence, gov)
                    event1 = "< " + " ".join(words_result).lower() + " >"

                    index_result, words_result, tag2 = old_extract_event_with_arg(sentence, dep)
                    event2 = "< " + " ".join(words_result).lower() + " >"

                    if tag1 == "verb_alone" or tag2 == "verb_alone":
                        continue

                    result_sentences.append({"doc_id": doc_id, "sentence": sentence, "event_pair": event1 + " " + connector + " " + event2})
    return result_sentences



def process_document(args, folder, newswire, idx, output_dir):

    for file_name in glob.glob(folder + "new_" + newswire + "_eng_*" + idx + ".txt"):
        #print file_name
        
        input_file = open(file_name, "r")
        found_sentences = []
        
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
                    if "prep" in args.pattern_type:
                        found_sentences += extract_sentences_nmod(sentenceList, doc_id)
                    if "advcl" in args.pattern_type:
                        found_sentences += extract_sentences_advcl(sentenceList, doc_id)
                continue
            if DOC_flag == True:
                document.append(line)
                continue

        input_file.close()

        output_sentences_file = output_dir + file_name.split("/")[-1].replace(".txt", "_sentences.txt")
        output_sentences = open(output_sentences_file, "w")

        sentence_str_set = set()
        for item in found_sentences:
            sentence = item["sentence"]
            sentence_str = " ".join(sentence.wordList).lower()
            if sentence_str in sentence_str_set:
                continue
            else:
                sentence_str_set.add(sentence_str)

            output_sentences.write("<doc_id> " + item["doc_id"] + "\n")
            output_sentences.write("<subevent> " + item["event_pair"] + "\n")
            output_sentences.write(" ".join(sentence.wordList) + "\n")
            output_sentences.write(" ".join(sentence.lemmaList) + "\n")
            output_sentences.write(" ".join(sentence.POSList) + "\n")
            output_sentences.write("|".join(sentence.dependencyList) + "\n\n\n")
        output_sentences.close()

        

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_type", dest="pattern_type", type=str, metavar='<str>', default='prep|advcl', help="Pattern type to use.")
    args = parser.parse_args()

    folder = "../../preprocess_gigaword/new_run/"
    #newswire_pool = ["nyt"]
    newswire_pool = ["afp", "apw", "cna", "ltw", "nyt", "wpb", "xin"]
    output_dir = "news/"
    

    for newswire in newswire_pool:
        print newswire
        processV = []
        for idx in range(0, 10):
            #output_sentences_file = output_dir + newswire + "_" + str(idx) + "_sentences.txt"
            processV.append(Process(target = process_document, args = (args, folder, newswire, str(idx), output_dir,)))
        for idx in range(0, 10):
            processV[idx].start()
            
        for idx in range(0, 10):
            processV[idx].join()

        print newswire, "finished!"