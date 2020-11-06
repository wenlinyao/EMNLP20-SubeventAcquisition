import sys
sys.path.append("../utilities/")
#reload(sys)
#sys.setdefaultencoding('utf-8')
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import networkx as nx
from nltk.corpus import wordnet as wn
import random, ast, time, os, math, copy, gensim, glob, nltk, pdb, argparse
#import cPickle as pickle
import pickle
from utilities import EventPair, extract_valid_pairs, extract_BERT_predicted_pairs, remove_brackets, light_verbs, pronouns, person_pronouns, extract_all_pairs
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA

#model = gensim.models.KeyedVectors.load_word2vec_format("../../tools/GoogleNews-vectors-negative300.bin", binary = True)


invalid_words = set(["person", "location", "be", "time", "act", "event", "activities",
                         "of", "for", "to", "up", "on", "with", "not", "at", "from", "into", "over", "by", "against","poss",
                         "about", "off", "before"])
invalid_words = invalid_words | light_verbs | pronouns | person_pronouns


def display_pca_scatterplot(words, trigger2vec, eval_iteration, flag):
    word_vectors = np.array([trigger2vec[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    fig = plt.figure(figsize=(12,12))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.00, y+0.00, word)
    fig.savefig(flag + "_" + str(eval_iteration)+'.png', dpi=fig.dpi)
    plt.show()
    fig.savefig(flag + "_" + str(eval_iteration)+'.pdf', bbox_inches='tight')


if __name__ == "__main__":

    words = ["conflict", "war", "attack", "protest", "clash", "fighting", "march", "game", "olympics", "match", \
             "bankruptcy", "reform", "recession", "investigation",\
             "hurricane", "storm", "earthquake", "flooding", "disaster",\
             "meeting", "conference", "forum", "discussion", \
             "festival", "ceremony", "celebration", \
             "election", "explosion", "wedding", "birthday", "carnival"] # "entertainment",
    input_lines = open("test_emb_20.txt", "r")

    trigger2vec = {}
    for line in input_lines:
        fields = line.split("\t")
        eventpair = EventPair(fields[0] + " -> " + fields[0], -1)
        if len(fields[0].split()) == 3:
            trigger2vec[eventpair.event1_trigger.replace("[","").replace("]","")] = ast.literal_eval(fields[1])

    input_lines.close()
    eval_iteration = 20
    display_pca_scatterplot(words, trigger2vec, eval_iteration, "child")
    





