import argparse
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="test_TruePred_5.txt", type=str, help="Evaluate on which file.")
    args = parser.parse_args()
    input_lines = open(args.file, "r")
    all_True_class = []
    all_Pred_class = []
    for line in input_lines:
        fields = line.split("\t")
        if " CONTAINS " in fields[0]:
            True_class = '1'
        elif " R_CONTAINS " in fields[0]:
            True_class = '2'
        else:
            True_class = fields[1]
        Pred_class = fields[2]

        all_True_class.append(True_class)
        all_Pred_class.append(Pred_class)
    print(all_True_class.count("0"), all_True_class.count("1"), all_True_class.count("2"))

    input_lines.close()
    labelList = ['0', '1', '2']
    results = precision_recall_fscore_support(all_True_class, all_Pred_class, average=None, labels=labelList)
    micro_avg = precision_recall_fscore_support(all_True_class, all_Pred_class, average='micro', labels=labelList[1:])
    results_str = []
    results_str.append("\t".join([str(l) for l in labelList]) + "\t" + "1_2_micro")
    results_str.append("\t".join(["%0.4f" % p for p in results[0]]) + "\t" + "%0.4f"%micro_avg[0])
    results_str.append("\t".join(["%0.4f" % r for r in results[1]]) + "\t" + "%0.4f"%micro_avg[1])
    results_str.append("\t".join(["%0.4f" % f for f in results[2]]) + "\t" + "%0.4f"%micro_avg[2])
    results_str = "\n".join(results_str)

    print(results_str)