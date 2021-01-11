from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys


######## NOT FINISHED (do not touch for now)#####################

def feature_extractor(c,feat,df):

    feature = df[df['sense'] == c]

    #print(feature[feat].tolist())

    return feature[feat].tolist()

def counter(classes, feature,df):
    counter_f= {}

    for c in classes:

        if c not in counter_f.keys():

                counter_f[c]={}
        print(c)

        out = feature_extractor(c,feature,df)
        print(out)

        for item in out:

            for i in item:

                if i not in counter_f[c].keys():
                    counter_f[c][i]= 0

                counter_f[c][i]+=1


    return counter_f

#########################################################

def pad(lst, n):
    cnt = 0
    while cnt < n:
        lst = np.insert(lst, 0, "x")
        cnt += 1
    # ~ print(lst)
    return list(lst)


def context_window(bucket_size, features):
    dat = pd.read_csv("dev.csv", index_col=0)

    out = pd.DataFrame(columns=['sentence', 'word', 'sense', 'context'])
    for f in features:
        out[f"{f}_context"] = ""

    for sent_id in dat['sentence'].unique():
        # get dataset for sentence and list the words in it
        sentence = dat[dat['sentence'] == sent_id]
        words = list(sentence['word'].values)

        # pad with "x"
        words = np.array(pad(words, (bucket_size - 1)/2))

        overlap_count = bucket_size - 1
        slider = Slider(bucket_size, overlap_count)
        slider.fit(words)
        while True:
            window_data = slider.slide()

            main_word = list(window_data)[int(overlap_count / 2)]
            sense = sentence[sentence['word'] == main_word]['sns'].values[0]

            # only consider words for "main word" when they have a sense
            if sense != 'O':
                stripped = list(np.delete(window_data, np.where(window_data == "x")))

                out = out.append(
                    pd.Series({'sentence': sent_id, 'word': main_word, 'sense': sense, 'context': stripped}),
                    ignore_index=True)

                for f in features:
                    
                    out[f"{f}_context"].loc[len(out[f"{f}_context"]) - 1] = list(
                        sentence[sentence['word'].isin(window_data)][f].values)

                    #print(out[f"{f}_context"].loc[len(out[f"{f}_context"]) - 1])

            if slider.reached_end_of_list(): break
    display(out)
    out.to_csv(f"context_size_{bucket_size}.csv")


if __name__ == '__main__':

    context_window(int(sys.argv[1]), ['sym', 'sem', 'sns'])

    #df = pd.read_csv (r'context_size_3.csv')
    
    #c=["male.n.02","carry.v.01"]
    #gen_feat=["sym_context","sem_context","sns_context"]


    #out=feature_extractor(c, gen_feat[0], df)
    #count= counter(c,gen_feat[0], df)
    #print(count)

