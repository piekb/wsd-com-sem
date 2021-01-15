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

    if c not in counter_f.keys():
        counter_f[c]={}

    for f in features:

        if f not in counter_f[c].keys():
            counter_f[c][f]= 0

        counter_f[c][f]+=1


    return counter_f

#########################################################


