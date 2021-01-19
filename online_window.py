import numpy as np
import pandas as pd
import sys
from IPython.display import display
from argparse import ArgumentParser
from count import *


def extract_dataset(file_name="data/csv/train.csv"):
    return pd.read_csv(file_name, index_col=0)


def eval(dataset_name, classify):

    df = extract_dataset(dataset_name)

    true_pos = 0
    total = 0

    for sent_id in df['sentence'].unique():

        # get dataset for sentence
        sentence = df[df['sentence'] == sent_id]

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

        for idx, (_, main_word) in enumerate(sentence.iterrows()):

            if main_word['sns'] == 'O':
                continue

            cls = main_word['sns']
            main_word['sns'] = None

            cls_features = feature_extractor(idx, sentence, features, bucket_size)
            counter_f = counter(cls, cls_features, counter_f)

            main_word['sns'] = cls

    return counter_f


if __name__ == '__main__':

    # Define parameters
    parser = ArgumentParser()

    parser.add_argument(
        '--feat',
        nargs = '+',
        type = str,
        required = True,
        help = 'Feature types to use',
    )

    parser.add_argument(
        '-k',
        type = float,
        default = 1.0,
        help = 'Laplacian smoothing constant',
    )

    parser.add_argument(
        '--bucket_sizes',
        type = int,
        nargs = '+',
        choices = [3, 5, 7, 9],
        required = True,
        help = 'Context window size',
    )

    parser.add_argument(
        '--pooling',
        type = str,
        default = 'average',
        choices = ['average', 'voting'],
        help = 'Pooling function to use for ensemble',
    )

    args = parser.parse_args()
    print('Using:')
    for k, v in args.__dict__.items():
        print(f'\t {k} : {v}')

    # Train model
    counter_fs = []
    for bucket_size in args.bucket_sizes:
        print(f'Training for bucket size={bucket_size}... ', end='')
        counter_fs.append(scrolling_window("data/csv/train.csv", bucket_size, args.feat))
        print('Done')

    # Create classifier
    if len(args.bucket_sizes) == 1:
        classify = make_naive_classify(args.feat, args.bucket_sizes[0], counter_fs[0], args.k)
    else:
        classify = make_ensemble_classify(args.feat, args.bucket_sizes, counter_fs, args.k, average_classifier if args.pooling == 'average' else voting_classifier)

    # Evaluate classifier
    acc_train = eval("data/csv/train.csv", classify)
    acc_dev = eval("data/csv/dev.csv", classify)

    print(f'Accuracy train {round(acc_train*100, 2)}%, dev {round(acc_dev*100, 2)}%')
