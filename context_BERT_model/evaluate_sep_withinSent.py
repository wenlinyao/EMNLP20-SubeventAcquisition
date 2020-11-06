import argparse, pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def evaluate_sep(labelList, filename2instanceList, file):
    pair2withinSent_flag = {}

    for filename in filename2instanceList:
        instanceList = filename2instanceList[filename]

        for instance in instanceList:
            if len(instance["masked_sentence"].split("[SEP]")) == 3:
                pair2withinSent_flag[instance["event_pair"]] = False
            elif len(instance["masked_sentence"].split("[SEP]")) == 2:
                pair2withinSent_flag[instance["event_pair"]] = True
            else:
                print("Error")

    input_lines = open(file, "r")

    all_True_class = []
    all_Pred_class = []

    all_True_class_within = []
    all_Pred_class_within = []

    all_True_class_cross = []
    all_Pred_class_cross = []

    for line in input_lines:
        fields = line.split("\t")

        True_class = fields[1]
        Pred_class = fields[2]

        all_True_class.append(True_class)
        all_Pred_class.append(Pred_class)

        if pair2withinSent_flag[fields[0]] == True:
            all_True_class_within.append(True_class)
            all_Pred_class_within.append(Pred_class)
        else:
            all_True_class_cross.append(True_class)
            all_Pred_class_cross.append(Pred_class)

    input_lines.close()

    results_str = []
    for item in [[all_True_class_within, all_Pred_class_within], [all_True_class_cross, all_Pred_class_cross], [all_True_class, all_Pred_class]]:
        all_True = item[0]
        all_Pred = item[1]
        results = precision_recall_fscore_support(all_True, all_Pred, average=None, labels=labelList)
        micro_avg = precision_recall_fscore_support(all_True, all_Pred, average='micro', labels=labelList[1:])
        
        results_str.append("\t".join([str(l) for l in labelList]) + "\t" + "1_2_micro")
        results_str.append("\t".join(["%0.4f" % p for p in results[0]]) + "\t" + "%0.4f"%micro_avg[0])
        results_str.append("\t".join(["%0.4f" % r for r in results[1]]) + "\t" + "%0.4f"%micro_avg[1])
        results_str.append("\t".join(["%0.4f" % f for f in results[2]]) + "\t" + "%0.4f"%micro_avg[2])
    results_str = "\n".join(results_str)
    print(results_str)
    return results_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="true_and_pred_value_6.txt", type=str, help="Evaluate on which file.")
    args = parser.parse_args()

    filename2instanceList = pickle.load(open("filename2instanceList.p", "rb"))
    evaluate_sep(filename2instanceList, args.file)
    
