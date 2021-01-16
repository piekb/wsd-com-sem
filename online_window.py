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


def extract_dataset(file_name="dev.csv"):
    return pd.read_csv(file_name, index_col=0)

def scrolling_window(dataset_name, bucket_size, features):

    df = extract_dataset(dataset_name)

    counter_f = {}

    for sent_id in df['sentence'].unique():

        # get dataset for sentence and list the words in it
        sentence = df[df['sentence'] == sent_id]
        words = list(sentence['word'].values)        
        words_idx = list(sentence.index)

        # pad with "x"
        words = np.array(pad_both(words, int((bucket_size - 1) / 2)))
        words_idx = np.array(pad_both(words_idx, int((bucket_size - 1) / 2)))

        if len(words) >= bucket_size:

            overlap_count = bucket_size - 1
            slider_words = Slider(bucket_size, overlap_count)
            slider_words.fit(words)

            slider_idx = Slider(bucket_size, overlap_count)
            slider_idx.fit(words_idx)

            while True:

                window_words = slider_words.slide()
                window_idx = list(map(lambda x : int(x), filter(lambda x : x!='x', slider_idx.slide())))
                window_data = sentence.loc[window_idx, :]


                # TODO match with sentences with padding on both sides
                main_word = list(window_words)[int(overlap_count / 2)]
                if main_word == "x": break
                sense = sentence[sentence['word'] == main_word]['sns'].values[0]

                # only consider words for "main word" when they have a sense
                if sense != 'O':
                    
                    extracted_features = feature_extractor(window_data, features)
                    counter_f = counter(sense, extracted_features, counter_f)

                if slider_words.reached_end_of_list(): break

    # display(out)
    # out.to_csv(f"context_size_{bucket_size}.csv")

    return counter_f


if __name__ == '__main__':


    
    gen_feat=["sym"]
    count= scrolling_window("train.csv", 3, gen_feat)