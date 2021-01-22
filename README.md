# wsd-com-sem
Code for Computational Semantics project: Implementation of an automated tagger for Word Sense Disambiguation (WSD). 

 Authors: M. Bouma (S3142558), F. Perin (S2865300)

## running experiments
To install the required packages, run:
`pip3 install -r requirements.txt`

Then run:
``python online_window.py --feat ('sym'|'sem'| 'sns')+ -k K --bucket_size (3| 5| 7)+ --pooling <voting|average>``

Note: do not use the --pooling parameter when --bucket_size has a single parameter. Furthermore, -k is a float value; we recommend assigning it between 0 and 10.

## data files
- txt: dev, train, and test files in the form of txt, from the conll data provided
- csv: dev, train, and test files in the form of csv, with cat and rol columns removed
- context: csv files of context windows from early version, based on dev

## online_window.py
Contains functions related to the scrolling context window, evaluation, and the main function of the code

## count.py
Contains functions related to counting frequencies, feature extraction from words in contexts, probabilities and calculation
