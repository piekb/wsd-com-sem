import numpy as np
import pandas as pd
import sys

# Function used to covert .txt file to .csv file using a dataframe. 
# Each line containce the ssentence ID, the word, the corresponding lemmatized word,
# the semantico role and the semantic derivations.
# Name of file needs to be manullay changed inside the code, separately from rest of the code.

def read_input(dataset_name):
    df = pd.DataFrame(columns = ["sentence", "word", "sym", "sem", "sns"])

    examples = []
    id = ''
    with open(dataset_name) as f:
        for line in f.readlines():
            if '#' in line:
                id = line.strip().split('=')[1]
            elif len(line) > 1:
                sp = line.strip().split('\t')
                df.loc[len(df)] = [id, sp[0], sp[1], sp[2], sp[4]]
    l=len(dataset_name)-4
    df.to_csv(dataset_name[:l]+'.csv')

if __name__ == '__main__':

    read_input("test.txt")

