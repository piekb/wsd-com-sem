# wsd-com-sem
Code for Computational Semantics project: Implementation of an automated tagger for Word Sense Disambiguation (WSD). 

Authors: M. Bouma (S3142558), F. Perin (S2865300)

To run the programs, run:
`pip3 install -r requirements.txt`

Then run:
`python3 online_window.py --feat -k --bucket_sizes --pooling` where 

- `--feat` is the set of features (sym/sem/sns)
- `-k` is the smoothing parameter (float)
- `--bucket_sizes` is the set of bucket sizes to be used (3/5/7/9)
- `--pooling` is the set of pooling functions to be used (average/voting)

## data files
- txt: dev, train, and test files in the form of txt, from the conll data provided
- csv: dev, train, and test files in the form of csv, with cat and rol columns removed
- context: csv files of context windows from early version, based on dev

## online_window.py
Contains functions related to the scrolling context window, evaluation, and the main function of the code

## count.py
Contains functions related to counting frequencies, feature extraction from words in contexts, probabilities and calculation
