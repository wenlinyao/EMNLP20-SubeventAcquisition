# EMNLP20-SubeventAcquisition
Code of the EMNLP2020 paper "Weakly Supervised Subevent Knowledge Acquisition"
Implementation is based on PyTorch 1.2

## Directory Description

subevent_pairs/: Acquired subevent knowledge

temporal_causal_pairs/: I also applied the proposed weakly supervised learning system to event temporal (before/after) or causal knowledge acquisition. The only difference is that I initialized the system with temporal patterns (i.e., before, after, then, follow) or causal patterns (lead, lead to, result, result in, result from, cause, due to, trigger, triggered by). Acquired event temporal knowledge and causal knowledge can be found here.

datasets/: Evaluation datasets that contain RED, ESC, HiEve, Timebank. Preprocessed data are in the name of [name]\_allRelations.txt. Only a few data instances are included for RED dataset considering the copyright of RED (https://catalog.ldc.upenn.edu/LDC2016T23). Please process the RED dataset into the data format by yourself.

dic/: All noun event trigger words.

distant_context_model/: Main system that uses weak supervision to extract subevent knowledge. The trained system will be automatically tested on RED dataset.

model/: Scripts to extract subevent pairs or collect sentences containing subevent pairs.

model_Trans/: TransE model to train event knowledge embeddings for evaluation. It will generate test_emb_20.txt (I trained 20 iterations). test_emb_20.txt storages vector representation of all event phrases that will be used in evaluation datasets (RED, Timebank, etc.). Each line is the vector representation of one event phrase.

context_BERT_model/: Evaluation models. Baseline BERT model together with two methods to incorporate subevent knowledge into BERT model.

Examples:
python ../context_BERT_model/BERT_main.py --preprocess True --eval_dataset HiEve --eval_relation subevent --gpu_id 0 --batch_size 16 --epochs 10 --sentence_setting across

python ../context_BERT_model/BERT_main.py --preprocess True --eval_dataset HiEve --eval_relation subevent --gpu_id 0 --batch_size 16 --epochs 10 --sentence_setting within

python ../context_BERT_model/BERT_main.py --preprocess True --eval_dataset Timebank --eval_relation temporal --gpu_id 0 --batch_size 16 --epochs 10 --sentence_setting across

python ../context_BERT_model/BERT_main.py --preprocess True --eval_dataset Timebank --eval_relation temporal --gpu_id 0 --batch_size 16 --epochs 10 --sentence_setting within

python ../context_BERT_model/BERT_main.py --preprocess True --eval_dataset ESL --eval_relation causal --gpu_id 0 --batch_size 16 --epochs 10 --sentence_setting across

python ../context_BERT_model/BERT_main.py --preprocess True --eval_dataset ESL --eval_relation causal --gpu_id 0 --batch_size 16 --epochs 10 --sentence_setting within

## All Event Relational Knowledge

If you are interested in entire event knowledge acquired by my previous papers. Please see this link: https://github.com/wenlinyao/EventCommonSenseKnowledge_dissertation

