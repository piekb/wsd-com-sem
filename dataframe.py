#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is just a visual thing, I don't like how Jupyter usually displays things
from IPython.display import display, HTML

CSS = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(CSS))


# # Data processing

# In[2]:


import pandas as pd
PARSING_EN_TRAIN = 'DRS_parsing-master/parsing/layer_data/gold/en/dev.conll'
df = pd.DataFrame(columns = ["sentence", "word", "sym", "sem", "sns"])

def read_input(in_file):
    examples = []
    id = ''
    with open("data.txt") as f:
        for line in f.readlines():
            if '#' in line:
                id = line.strip().split('=')[1]
            elif len(line) > 1:
                sp = line.strip().split('\t')
                df.loc[len(df)] = [id, sp[0], sp[1], sp[2], sp[4]]
    return df

dat = read_input(PARSING_EN_TRAIN)
display(dat)


# # Count occurrences of each word, respective to each word sense
# ### Make dataframe of all unique words and their total counts (not respective to sense)

# In[3]:


unique_words = dat['word'].value_counts().rename_axis('unique_word').reset_index(name='total_count')
display(unique_words)


# ### Per unique word, get all unique senses it's used with and their counts.  Store in another dataframe. Also, store all ambiguous words in another dataframe. 

# In[4]:


ambiguous_words = pd.DataFrame(columns = ['word', 'sense', 'count', 'normalize'])
word_df = pd.DataFrame(columns = ['word', 'sense', 'count', 'normalize' ])

# Loop over unique words and get their unique senses
for word in unique_words['unique_word']:
    ambiguous = False
    word_dis = dat[dat['word'] == word]
    sense_counts = word_dis['sns'].value_counts().rename_axis('sense').reset_index(name='counts')
    norm= word_dis['sns'].value_counts(normalize=True).rename_axis('sense').reset_index(name='normalize')
    #display(sense_counts)
    #display(norm)
    new=pd.merge(sense_counts, norm, on=['sense'], how='left')
    #display(norm)
    if len(sense_counts) > 1:
        ambiguous = True
    
    # Store in dataframe
    for i, row in new.iterrows():
        word_df.loc[len(word_df)] = [word, row['sense'], row['counts'], row['normalize']]
        if ambiguous:
            ambiguous_words.loc[len(ambiguous_words)] = [word, row['sense'], row['counts'], row['normalize']]
            
display(word_df)
display(ambiguous_words)


# In[10]:


# Some examples
display(word_df[word_df['word']=='Tom'])
display(word_df[word_df['word']=='is'])


# # Naive Bayesian classifier

# In[33]:


# To choose viable window sizes, find some statistics on sentence length
# NOTE: so far, this is including punctuation! 
lengths = dat['sentence'].value_counts().rename_axis('sentence').reset_index(name='length')
display(lengths)
display(lengths['length'].describe())


# In[34]:


display(lengths['length'].plot.hist())


# In[ ]:




