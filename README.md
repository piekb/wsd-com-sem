# wsd-com-sem
Code for Computational Semantics project on Word Sense Disambiguation (WSD). 

Authors: M. Bouma (S3142558), F. Perin (S2865300) at MSc AI, Rijksuniversiteit Groningen

## running experiments
To install the required packages, run:
`pip install -r requirements.txt`

Then run:
``python online_window.py --feat ('sym'|'sem'| 'sns')+ -k K --bucket_size (3| 5| 7)+ --pooling <voting|average>``

Notes: 
- do not use the --pooling parameter when --bucket_size has a single parameter. 
- -k is a float value; we recommend assigning it between 0 and 10. 
- adjust pip and python if needed to use Python 3.

## data files
- txt: dev, train, and test files in the form of txt, from the conll data provided.
- csv: dev, train, and test files in the form of csv, with cat and rol columns removed.
- context: csv files of context windows from early version, based on dev.

## dev_files
Files, code, notebooks, etc. used for development that are not relevant to any other user. 

## data_exp files
- data_exp.ipynb: jupyter notebook describing data exploration.
- data_exp.html: html version of the notebook for easier viewing. 

## online_window.py
Contains functions related to the scrolling context window, evaluation, and the main function of the code.

## count.py
Contains functions related to counting frequencies, feature extraction from words in contexts, probabilities and calculation.