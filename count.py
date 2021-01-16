from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys


######## NOT FINISHED (do not touch for now)#####################

def feature_extractor(sentence, feat):

    l=[]


    for _, word_data in sentence.iterrows():
        for feat_name in feat:
            l.append(feat_name + '_' + word_data[feat_name])

    return l

def log_probs(super_class, features, counter_f, k):

    probs = dict()

    tot_cls_freq = 0
    for c in counter_f[super_class]['cls'].values():
        tot_cls_freq += c['tot']

    for c in counter_f[super_class]['cls'].keys():

        # Add prior
        probs[c] = np.log(c['tot']/tot_cls_freq)


        # Sum log probs features
        for f in features:

            if k!=0 or (f in c['feats'].keys()) :
                probs[c] += np.log((c['feats'][f]+k)/(c['tot']+ len(counter_f[super_class]['vocab'])*k))

    return probs

def greedy_classify(class_probs):

    best = None

    for c, c_prob in class_probs.items():

        if best is None or class_probs[best] < c_prob:
            best = c

    return best


def counter(c, features, counter_f = {}):

    super_class = '.'.join(c.split('.')[:2])

    if super_class not in counter_f.keys():
        counter_f[super_class] = dict(
            vocab = [],
            cls = dict()
        )

    if c not in counter_f[super_class].keys():
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


