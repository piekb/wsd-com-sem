from IPython.display import display
import numpy as np
import pandas as pd
import sys
import random


# Extracts the given features for words in the given sentence
def feature_extractor(idx, sentence, feat, bucket_size):

    sentence = sentence[max(0, idx-bucket_size//2):min(len(sentence), idx+bucket_size//2)]

    l=[]

    for _, word_data in sentence.iterrows():
        for feat_name in feat:

            if word_data[feat_name] is not None:
                l.append(feat_name + '_' + word_data[feat_name])

    return l


def log_probs(super_class, features, counter_f, k):

    probs = dict()

    if super_class not in counter_f:
        return None

    tot_cls_freq = 0
    for c in counter_f[super_class]['cls'].values():
        tot_cls_freq += c['tot']

    for c, c_freq in counter_f[super_class]['cls'].items():

        # Add prior
        probs[c] = np.log(c_freq['tot']/tot_cls_freq)

        # Sum log probs features
        for f in features:

            if k!=0 or (f in c_freq['feats'].keys()) :
                probs[c] += np.log((c_freq['feats'].get(f, 0)+k)/(c_freq['tot']+ len(counter_f[super_class]['vocab'])*k))

    return probs


def greedy_classifier(class_probs):

    if class_probs is None:
        return 'unknown'

    best = None

    for c, c_prob in class_probs.items():

        if best is None or class_probs[best] < c_prob:
            best = c

    return best


def voting_classifier(class_probs_list):

    classes = list(map(greedy_classifier, class_probs_list))

    votes = {}
    answer = None

    for cls in classes:
        votes[cls] = votes.get(cls, 0) + 1

        if answer is None or votes[answer]<votes[cls]:
            answer = cls

    best_classes = list(
        map(
            lambda x : x[0],
            filter(
                lambda x : x[1]==votes[answer],
                votes.items(),
            ),
        )
    )

    return random.sample(best_classes, 1)[0]


def average_classifier(class_probs_list):

    probs={}
    empty=True

    for class_probs in class_probs_list:
        if class_probs is not None:
            empty=False
            break

    if empty:
        return 'unknown'

    for cls in class_probs.keys():
        prob=0

        for item in class_probs_list:
            prob+=item[cls]

        probs[cls]=prob/len(class_probs_list)

    return greedy_classifier(probs)


def make_naive_classify(feat, bucket_size, counter_f, k):

    def classify(idx, sentence, super_class):

        features = feature_extractor(idx, sentence, feat, bucket_size)
        class_probs = log_probs(super_class, features, counter_f, k)

        return greedy_classifier(class_probs)

    return classify


def make_ensemble_classify(feat, bucket_sizes, counter_fs, k, pooling_fn):

    def classify(idx, sentence, super_class):

        class_probs_list=[]

        for bucket_size, counter_f in zip(bucket_sizes, counter_fs):

            features = feature_extractor(idx, sentence, feat, bucket_size)

            class_probs_list.append(log_probs(super_class, features, counter_f, k))

        return pooling_fn(class_probs_list)

    return classify


def counter(c, features, counter_f = {}):

    super_class = '.'.join(c.split('.')[:2])

    if super_class not in counter_f.keys():
        counter_f[super_class] = dict(
            vocab = [],
            cls = dict()
        )

    if c not in counter_f[super_class]['cls'].keys():
        counter_f[super_class]['cls'][c] = dict(
            tot = 0,
            feats = dict()
        )

    for f in features:

        if f not in counter_f[super_class]['vocab']:
            counter_f[super_class]['vocab'].append(f)

        if f not in counter_f[super_class]['cls'][c]['feats'].keys():
            counter_f[super_class]['cls'][c]['feats'][f]= 0

        counter_f[super_class]['cls'][c]['feats'][f]+=1
        counter_f[super_class]['cls'][c]['tot'] +=1

    return counter_f
