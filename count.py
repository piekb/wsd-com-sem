from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys

######## NOT FINISHED (do not touch for now)#####################

def feature_extractor(c, feat, df):

    l=[]
    feature = df[df['sense'] == c]

    #print(feature[feat].tolist())

    for item in feature.loc[:, feat].values:

        for feat_name, feat_vals in zip(feat, item):

            cont_elem= list(feat_vals.strip("] [").replace("'", '').split(", "))

            for i in cont_elem:

                l.append(feat_name + '_' + i)

    return l

def counter(classes, feature, df):
    counter_f= {}

    for c in classes:

        counter_f[c]={}

        out = feature_extractor(c, feature, df)

        for f in out:

            if f not in counter_f[c].keys():
                counter_f[c][f]= 0

            counter_f[c][f]+=1


    return counter_f

#########################################################

if __name__ == '__main__':

    df = pd.read_csv (r'context_size_3.csv')

    print(df)
    
    c=["male.n.02","carry.v.01"]
    gen_feat=["sym_context","sem_context","sns_context"]


    #out=feature_extractor(c, gen_feat[0], df)
    count= counter(c,gen_feat, df)
    print(count)
