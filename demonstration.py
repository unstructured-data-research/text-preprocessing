#%%

import pandas as pd
import numpy as np
import sys
import string 

# load our own preprocessing module from the src folder
sys.path.append('src')
import preprocessing_class as pc

input_path = "./data/"
# %%

df = pd.read_csv(input_path + "mpc_minutes.txt", sep = "\t")
df
# %%

#=============
# Preprocess data
#=============

# define dictionary for preprocessing class with terms we want to preserve
replacing_dict = {'monetary policy':'monetary-policy',
                  'interest rate':'interest-rate',
                  'interest rates':'interest-rate',
                  'yield curve':'yield-curve',
                  'repo rate':'repo-rate',
                  'bond yields':'bond-yields',
                  'real estate':'real-estate',
                  'economic growth':'economic-growth',
                  "long term interest rates": "long-term-interest-rates"}

# define tokenization pattern and punctuation symbols
pattern = r'''
          (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
          \w+(?:-\w+)*        # word characters with internal hyphens
          | [][.,;"'?():-_`]  # preserve punctuation as separate tokens
          '''
punctuation = string.punctuation.replace("-", "")

#%%

# initialize the class with the text data and some parameters
prep = pc.RawDocs(df["minutes"])

# replace specific phrases of interest
prep.phrase_replace(replace_dict=replacing_dict,
                    sort_dict=True,
                    case_sensitive_replacing=False)

# lower-case text, expand contractions and initialize stopwords list
prep.basic_cleaning(lower_case=True,
                    contraction_split=True)

# split the documents into tokens
prep.tokenize_text(tokenization_pattern=pattern)

# clean tokens (remove non-ascci characters, remove short tokens, remove punctuation and numbers)
prep.token_clean(length=2, 
                 punctuation=punctuation, 
                 numbers=True)

# remove stopwords
prep.stopword_remove(items='tokens', stopwords="short")

# create document-term matrix (can use the min_df parameter to remove words that appear in less than min_df documents)
prep.dt_matrix_create(items='tokens', min_df=1, score_type='df')

# get the vocabulary and the appropriate dictionaries to map from indices to words
word2idx = prep.vocabulary["tokens"]
idx2word = {i:word for word,i in word2idx.items()}
vocab = list(word2idx.keys())
print("Vocabulary size: ", len(vocab))

#%%

# inspect a particular tokenized document and compare it to its original form
i = 4
print(df.loc[i, "minutes"])
print("\n ------------------------------- \n")
print(prep.tokens[i])
# %%
