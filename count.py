from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys


######## NOT FINISHED (do not touch for now)#####################

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

def greedy_classify(class_probs):

    if class_probs is None:
        return 'unknown'

    best = None

    for c, c_prob in class_probs.items():

        if best is None or class_probs[best] < c_prob:
            best = c

    return best

def make_naive_classify(feat, bucket_size, counter_f, k):

    def classify(idx, sentence, super_class):

        features = feature_extractor(idx, sentence, feat, bucket_size)
        class_probs = log_probs(super_class, features, counter_f, k)

        return greedy_classify(class_probs)

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

#########################################################


