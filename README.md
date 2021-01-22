# wsd-com-sem
Code for Computational Semantics project: Implementation of an automated tagger for Word Sense Disambiguation (WSD). 

Authors: M. Bouma (S3142558), F. Perin (S2865300)

## dataframe files
- .ipynb: main notebook
- .py: python copy of main notebook
- .html: link to visual version of notebook without having to run jupyter notebook

## data files
- data.txt: txt file of dev.conll data
- dev.csv: csv file of dev.conll data
- context_size_3.csv: dataset of context window size 3

## window file
- window.py: takes window size as argument to produce dataset with context frames

## running experiments

``python online_window.py --feat ('sym'|'sem'| 'sns')+ -k K --bucket_size (3| 5| 7)+ --pooling <voting|average>``

do not use the --pooling parameter when --bucket_size has a single parameter.

Furthermore K is a float value (we reccomend assignining it between 0 and 10)