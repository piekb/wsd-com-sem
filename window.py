from window_slider import Slider
from IPython.display import display
import numpy as np
import pandas as pd
import sys


def pad_both(lst, pad_size):
    before = pad_size * ["x"]
    after = (pad_size - 1) * ["x"]
    lst = before + lst + after
    print(lst)
    return lst


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
        words = np.array(pad_both(words, int((bucket_size - 1) / 2)))
        if len(words) >= bucket_size:
            overlap_count = bucket_size - 1
            slider = Slider(bucket_size, overlap_count)
            slider.fit(words)
            while True:
                window_data = slider.slide()

                # TODO match with sentences with padding on both sides
                main_word = list(window_data)[int(overlap_count / 2)]
                if main_word == "x": break
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

                if slider.reached_end_of_list(): break
        else:
            print(words, len(words))
    display(out)
    out.to_csv(f"context_size_{bucket_size}.csv")


if __name__ == '__main__':
    context_window(int(sys.argv[1]), ['sym', 'sem', 'sns'])
    #s = ['Tom', 'was', 'carrying', 'a', 'bucket', 'of', 'water', '.']
    #s2 = ['Hey', '!']
    #s3 = ['The', 'weather']
    #pad_both(s3, 2)
