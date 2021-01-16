from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys

from count import *

def pad_both(lst, pad_size):
    before = pad_size * ["x"]
    after = (pad_size - 1) * ["x"]
    lst = before + lst + after
    # print(lst)
    return lst


def pad(lst, n):
    cnt = 0
    while cnt < n:
        lst = np.insert(lst, 0, "x")
        cnt += 1
    # ~ print(lst)
    return list(lst)


def extract_dataset(file_name="data/csv/train.csv"):
    return pd.read_csv(file_name, index_col=0)

def scrolling_window(dataset_name, bucket_size, features):

    df = extract_dataset(dataset_name)

    counter_f = {}

    for sent_id in df['sentence'].unique():

        # get dataset for sentence and list the words in it
        sentence = df[df['sentence'] == sent_id]
        # words = list(sentence['word'].values)        
        # words_idx = list(sentence.index)

        # pad with "x"
        # words = np.array(pad_both(words, int((bucket_size - 1) / 2)))
        # words_idx = np.array(pad_both(words_idx, int((bucket_size - 1) / 2)))

        for idx, main_word in sentence.iterrows():

            if main_word['sns'] == 'O':
                continue

            cls = main_word['sns']
            main_word['sns'] = None

            cls_features = feature_extractor(idx, sentence, features, bucket_size)
            counter_f = counter(cls, cls_features, counter_f)

            main_word['sns'] = cls

    # display(out)
    # out.to_csv(f"context_size_{bucket_size}.csv")

    return counter_f


if __name__ == '__main__':


    
    gen_feat=["sym", "sns"]
    count= scrolling_window("data/csv/train.csv", 5, gen_feat)