import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

"""
######### with knowledge vector ##########
0   1   2
0.9458  0.5625  0.5789
0.9769  0.3103  0.4231
0.9611  0.4000  0.4889
"""
def micro_cross_val(args, fileList):
    all_True = []
    all_Pred = []

    for file in fileList:
        input_file = open(file, "r")
        for line in input_file:
            fields = line.split("\t")
            True_class = fields[1]
            Pred_class = fields[2]
            
            if args.eval_direction == False:
                if True_class in ["1", "2"]:
                    True_class = "1"
                if Pred_class in ["1", "2"]:
                    Pred_class = "1"
            
            all_True.append(True_class)
            all_Pred.append(Pred_class)
        input_file.close()

    labelList = [str(label) for label in range(0, args.class_num)]

    #labelList = ["0", "1", "2"]
    results = precision_recall_fscore_support(all_True, all_Pred, average=None, labels=labelList)
    micro_avg = precision_recall_fscore_support(all_True, all_Pred, average='micro', labels=labelList[1:])
    results_str = []
    results_str.append("\t".join([str(l) for l in labelList]) + "\t" + "1_2_micro")
    results_str.append("\t".join(["%0.4f" % p for p in results[0]]) + "\t" + "%0.4f"%micro_avg[0])
    results_str.append("\t".join(["%0.4f" % r for r in results[1]]) + "\t" + "%0.4f"%micro_avg[1])
    results_str.append("\t".join(["%0.4f" % f for f in results[2]]) + "\t" + "%0.4f"%micro_avg[2])

    table_str = ""

    for i in range(0, len(results[0])):
        table_str += "%0.2f"%(results[0][i]*100)+"/"+"%0.2f"%(results[1][i]*100)+"/"+"%0.2f"%(results[2][i]*100) + " "

    results_str.append(table_str + " " + "%0.2f"%(micro_avg[0]*100)+"/"+"%0.2f"%(micro_avg[1]*100)+"/"+"%0.2f"%(micro_avg[2]*100))
    results_str = "\n".join(results_str)

    return results_str

def read_perf_lines(perf_lines):
    assert len(perf_lines) == 4 * 3
    # skip one line
    PList_within = [float(p) for p in perf_lines[1].split()]
    RList_within = [float(r) for r in perf_lines[2].split()]
    FList_within = [float(f) for f in perf_lines[3].split()]

    # skip one line
    PList_cross = [float(p) for p in perf_lines[5].split()]
    RList_cross = [float(r) for r in perf_lines[6].split()]
    FList_cross = [float(f) for f in perf_lines[7].split()]

    # skip one line
    PList_all = [float(p) for p in perf_lines[9].split()]
    RList_all = [float(r) for r in perf_lines[10].split()]
    FList_all = [float(f) for f in perf_lines[11].split()]

    perf_matrix = [PList_within, RList_within, FList_within, PList_cross, RList_cross, FList_cross, PList_all, RList_all, FList_all]
    return perf_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_direction", default=True, type=str2bool, help="Consider direction or not (e1->e2 or e1<-e2).")
    parser.add_argument("--eval_file", default="results_strList.txt", type=str, help="Performance file to read.")
    parser.add_argument("--eval_epoch", default=6, type=int, help="Which epoch to use for evaluation.")
    parser.add_argument("--cross_folds", default=5, type=int, help="Cross-validation folds (default 5).")
    parser.add_argument("--class_num", default=3, type=int, help="Class number.")
    args = parser.parse_args()
    perfList1 = []
    perfList2 = []
    perfList3 = []
    perfList4 = []
    input_file = open(args.eval_file, "r")
    perf_lines = []
    for line in input_file:
        if len(line.split()) == 0 and len(perf_lines) != 0:
            #print(perf_lines)
            perf_matrix = read_perf_lines(perf_lines)
            if flag == "1":
                perfList1.append(perf_matrix)
            elif flag == "2":
                perfList2.append(perf_matrix)
            elif flag == "3":
                perfList3.append(perf_matrix)
            elif flag == "4":
                perfList4.append(perf_matrix)
            perf_lines = []
            continue
        if "#####" in line:
            if "knowledge_vector = False, children_vector = False" in line:
                flag = "1"
            elif "knowledge_vector = True, children_vector = False" in line:
                flag = "2"
            elif "knowledge_vector = False, children_vector = True" in line:
                flag = "3"
            elif "knowledge_vector = True, children_vector = True" in line:
                flag = "4"
            continue
        perf_lines.append(line)

    input_file.close()
    #print(with_perfList)


    epochs = max(len(perfList1), len(perfList2), len(perfList3), len(perfList4)) // args.cross_folds
    print("Epochs:", epochs)

    if len(perfList1) != 0:
        assert len(perfList1) >= args.cross_folds * epochs
        target_epochs = args.eval_epoch
        #target_epochs = 6
        perf = []
        fileList = []
        for i in range(0, args.cross_folds):
            perf1 = perfList1[i*epochs+target_epochs-1]
            perf.append(perf1)
            fileList.append(str(i) + "_S1_true_and_pred_value_" + str(target_epochs-0) + ".txt")
            
        avg_perf = np.sum(perf, axis=0) / float(args.cross_folds)
        print("knowledge_vector = False, children_vector = False")
        #print(avg_perf)
        print(micro_cross_val(args, fileList))

    if len(perfList2) != 0:
        assert len(perfList2) >= args.cross_folds * epochs
        target_epochs = args.eval_epoch
        perf = []
        fileList = []
        for i in range(0, args.cross_folds):
            perf1 = perfList2[i*epochs+target_epochs-1]
            perf.append(perf1)
            fileList.append(str(i) + "_S2_true_and_pred_value_" + str(target_epochs-0) + ".txt")
            
        avg_perf = np.sum(perf, axis=0) / float(args.cross_folds)
        print("knowledge_vector = True, children_vector = False")
        #print(avg_perf)
        print(micro_cross_val(args, fileList))

    if len(perfList3) != 0:
        assert len(perfList3) >= args.cross_folds * epochs
        target_epochs = args.eval_epoch
        perf = []
        fileList = []
        for i in range(0, args.cross_folds):
            perf1 = perfList3[i*epochs+target_epochs-1]
            perf.append(perf1)
            fileList.append(str(i) + "_S3_true_and_pred_value_" + str(target_epochs-0) + ".txt")
            
        avg_perf = np.sum(perf, axis=0) / float(args.cross_folds)
        print("knowledge_vector = False, children_vector = True")
        #print(avg_perf)
        print(micro_cross_val(args, fileList))

    if len(perfList4) != 0:
        assert len(perfList4) >= args.cross_folds * epochs
        target_epochs = args.eval_epoch
        perf = []
        fileList = []
        for i in range(0, args.cross_folds):
            perf1 = perfList4[i*epochs+target_epochs-1]
            perf.append(perf1)
            fileList.append(str(i) + "_S4_true_and_pred_value_" + str(target_epochs-0) + ".txt")
            
        avg_perf = np.sum(perf, axis=0) / float(args.cross_folds)
        print("knowledge_vector = True, children_vector = True")
        #print(avg_perf)
        print(micro_cross_val(args, fileList))
    
    