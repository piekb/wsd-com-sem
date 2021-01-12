from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys

######## NOT FINISHED (do not touch for now)#####################

def clean_up_str(s):

    if s.startswith(' '):
        s=s[:1]

    return s

def feature_extractor(c,feat,df):

    feature = df[df['sense'] == c]

    #print(feature[feat].tolist())

    return feature[feat]

def counter(classes, feature,df):
    counter_f= {}

    for c in classes:

        if c not in counter_f.keys():

            counter_f[c]={}
        print(c)

        out = feature_extractor(c,feature,df)
        print(out)

        for item in out:

            list_item = list(item.strip("] [").split(", "))

            print(list_item)

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
    count= counter(c,gen_feat[0], df)
    print(count)
