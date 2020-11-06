import pickle, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_num", default=3, type=int, help="Class number.")
    args = parser.parse_args()

    filename2instanceList = pickle.load(open("filename2instanceList.p", "rb"))
    within_classList = []
    across_classList = []
    for filename in filename2instanceList:
        for instance in filename2instanceList[filename]:
            if len(instance["masked_sentence"].split("[SEP]")) == 3:
                across_classList.append(instance["class"])
            else:
                within_classList.append(instance["class"])

    print("within: ", list(range(0, args.class_num)))
    for i in range(0, args.class_num):
        print(within_classList.count(i),)

    print("across: ", list(range(0, args.class_num)))
    for i in range(0, args.class_num):
        print(across_classList.count(i),)