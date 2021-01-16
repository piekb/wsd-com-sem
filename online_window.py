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

def eval(dataset_name, classify):

    df = extract_dataset(dataset_name)

    true_pos = 0
    total = 0

    for sent_id in df['sentence'].unique():

        # get dataset for sentence and list the words in it
        sentence = df[df['sentence'] == sent_id]
        # words = list(sentence['word'].values)        
        # words_idx = list(sentence.index)

        # pad with "x"
        # words = np.array(pad_both(words, int((bucket_size - 1) / 2)))
        # words_idx = np.array(pad_both(words_idx, int((bucket_size - 1) / 2)))

        for idx, (_, main_word) in enumerate(sentence.iterrows()):

            if main_word['sns'] == 'O':
                continue

            cls = main_word['sns']
            main_word['sns'] = None

            super_class = '.'.join(cls.split('.')[:-1])

            cls_predicted = classify(idx, sentence, super_class)

            true_pos += int(cls == cls_predicted)
            total += 1

            main_word['sns'] = cls

    return true_pos/total

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

        for idx, (_, main_word) in enumerate(sentence.iterrows()):

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

    # Define parameters
    feat=["sym", "sns"]
    k = 1
    bucket_size = 3

    # Train model
    print('Training... ', end='')
    counter_f = scrolling_window("data/csv/train.csv", bucket_size, feat)
    print('done')

    # Create classifier
    classify = make_naive_classify(feat, bucket_size, counter_f, k)

    # Evaluate classifier
    acc_train = eval("data/csv/train.csv", classify)
    acc_dev = eval("data/csv/dev.csv", classify)

    print(f'Accuracty train {round(acc_train*100, 2)}%, dev {round(acc_dev*100, 2)}%')