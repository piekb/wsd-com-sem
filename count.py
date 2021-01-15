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

def counter(c, features, counter_f = {}):

    super_class = '.'.join(c.split('.')[:2])

    if super_class not in counter_f.keys():
        counter_f[super_class] = {}

    if c not in counter_f[super_class].keys():
        counter_f[super_class][c] = dict(
            tot = 0,
            feats = dict()
        )


    counter_f[super_class][c]['tot'] +=1

    for f in set(features):

        if f not in counter_f[super_class][c]['feats'].keys():
            counter_f[super_class][c]['feats'][f]= 0

        counter_f[super_class][c]['feats'][f]+=1


    return counter_f

#########################################################


