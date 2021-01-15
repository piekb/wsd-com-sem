from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys

######## NOT FINISHED (do not touch for now)#####################

def feature_extractor(c, feat, df):

    feature = df[df['sense'] == c]

    #print(feature[feat].tolist())

    return feature.loc[:, feat]

def counter(classes, feature, df):
    counter_f= {}

    for c in classes:

        counter_f[c]={}

        out = feature_extractor(c, feature, df)

        for item in out.values:

            for feat_type in item:

                list_item = list(feat_type.strip("] [").split(", "))

                for i in list_item:

                    if i not in counter_f[c].keys():
                        counter_f[c][i]= 0

                    counter_f[c][i]+=1


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
