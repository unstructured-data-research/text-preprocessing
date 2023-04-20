# %%
from __future__ import division
import sys
sys.path.append('/opt/conda/lib/python3.7/site-packages')

# sys.path
# %%

#####################################  Functions #################################### 
import string
import unicodedata
import itertools
import re
import numpy as np
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk import SnowballStemmer
stemmer = SnowballStemmer(language='english')

from flashtext import KeywordProcessor
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex


nlp_standard = spacy.load('en_core_web_sm')
# %%
#modify spacy tokenizer to not split "word-word" when lemmatizing
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp_lemmatizer = spacy.load('en_core_web_sm')
nlp_lemmatizer.tokenizer.infix_finditer = infix_re.finditer

# contains stopwords
import preprocess_data
#%%
class RawDocs():

    def __init__(self, doc_data):

        """

        Initialize the class with the text data.

        Arguments:
        ----------
        - doc_data: (1) path to text file with each document on new line, or
          (2) Python iterable of strings. Strings should have utf-8 encoded
          characters.

        """

        if isinstance(doc_data, str):
            print(f"Reading data from {doc_data}")
            raw = open(doc_data, encoding="utf-8").read()
            self.docs = raw.splitlines()
        else:
            #iterator = iter(doc_data)
            print("Reading data from iterator")
            try:
                self.docs = [s.encode('utf-8').decode('utf-8')
                             for s in doc_data]
            except UnicodeDecodeError:
                print("At least one string does not have utf-8 encoding")


    def basic_cleaning(self, lower_case=True, contraction_split=True):
        
        """
        Basic cleaning of documents.

        Arguments:
        ----------

        - lower_case: defaults to True, converts the whole corpus to lower case
        - contraction_split: defaults to True, splits contractions into constituent words
          If False, remove all apostrophes.
        
        """
        
        # save parameters as class attributes
        self.lower_case = lower_case
        self.contraction_split = contraction_split

        # lower-case
        if self.lower_case:
            self.docs = [s.lower() for s in self.docs]

        # split contractions
        if self.contraction_split:
            self.docs = list(map(lambda x: contractions.fix(x), self.docs))
        else:
            self.docs = list(map(lambda x: re.sub(u'[\u2019\']', '', x), self.docs)) 

        # save the number of documents as an attribute
        self.N = len(list(self.docs)) 


    def tokenize_text(self, tokenization_pattern=None):
        """
        Split documents into tokens.
        
        Arguments:
        ----------
        - tokenization_pattern: defaults to None, simple NLTK tokenizer is used (mind that by default it
          splits over "-"). Else, a customized regex pattern can be passed to modify the tokenizer.
        
        """

        # save parameters as class attributes
        self.tokenization_pattern = tokenization_pattern

        if self.tokenization_pattern is None:
            # define a basic tokenization strategy
            def spacy_tokenizer_standard(sent):
                sent = nlp_standard.tokenizer(sent)
                tokens = [t.text for t in sent]
                return tokens
            self.tokens = list(map(spacy_tokenizer_standard, self.docs))
        else:
            self.tokens = list(map(lambda x:nltk.regexp_tokenize(x,
                                                                 pattern=self.tokenization_pattern), self.docs))

    def get_named_entities(self):
        '''
        Produce a tentative list of named entities via spaCy named entity recognition algorithm 
        applied to \'tokens\'. The result is stored in \'named_entities\'
        '''
        def get_ents_sent(sent):
                sent = ' '.join(sent)
                doc = nlp_lemmatizer(sent)
                return list(doc.ents)
        self.named_entities = [e.text for el in map(get_ents_sent, self.tokens) for e in el] 

    def phrase_replace(self, replace_dict, sort_dict=True, case_sensitive_replacing=False):

        """
        Replace terms/phrases according to mapping defined in replace_dict
        
        If ngrams, the user should join the terms with a consistent character (e.g. '_', '-' )
        and mind to drop the chosen joining character from the punctuation to be removed in token_clean().
        
        Keys in the replace_dict should will be sorted by the number of terms they contain and
        replacement will be done in that order.

        One can pass user-defined list of words to remove by replacing with ''.
        
        Arguments:
        ----------

        replace_dict: dictionary with terms/phrases to be replaced as keys and the replacement as values
        sort_dict: defaults to True, if True sorts the dictionary by the number of terms in the key and by their length
        case_sensitive_replacing: defaults to False, if True allows the replacement to be case sensitive
        
        """

        # sort the dictionary by the number of terms in the key and by their length
        sorted_replace_dict = {k: v for k, v in sorted(replace_dict.items(), key=lambda item: (len(item[0].split()), len(item[0])), reverse=True)}

        if sort_dict:
            replace_dict = sorted_replace_dict
        else:
            # check if the order of the keys in the provided dictionary is the same as the sorted one
            # using map() & all() to check if both the lists are exactly equal
            if not all(map(lambda x, y: x == y, replace_dict.keys(), sorted_replace_dict.keys())):
                print("Warning: the provided dictionary is not sorted. Replacement will be done in the order of the keys provided by the user. If you want to sort the dictionary, set sort_dict=True.")

        # add all the elements of the replacement dictionary into the KeywordProcessor
        keyword_processor = KeywordProcessor(case_sensitive=case_sensitive_replacing)
        keyword_processor.set_non_word_boundaries(set(string.digits + string.ascii_letters + '-'))
        for k,v in replace_dict.items():
            keyword_processor.add_keyword(k, v)

        # apply to all documents and update them
        self.docs = list(map(keyword_processor.replace_keywords, self.docs))

        # save replacement dictionary as an attribute
        self.replace_dict = replace_dict


    def make_tokens_lowercase(self):
        """ Lowercase all tokens in documents
        """
        tokens_lower = []
        for i, doc in enumerate(self.tokens):
            doc_lower = [w.lower() for w in doc]
            tokens_lower.append(doc_lower)
        
        self.tokens = tokens_lower
    
    def bigram(self, items, joining_char='-'): 

        """
        Generate bigrams of either items = "tokens", "lemmas" or "stems"
        """

        def bigram_join(tok_list):
            text = nltk.bigrams(tok_list)
            return list(map(lambda x: x[0] + joining_char + x[1], text))

        if items == "tokens":
            self.bigrams = list(map(bigram_join, self.tokens))
        elif items == "lemmas":
            self.bigrams = list(map(bigram_join, self.lemmas))
        elif items == "stems":
            self.bigrams = list(map(bigram_join, self.stems))
        else:
            raise ValueError("Items must be either \'tokens\', \'lemmas\' or \'stems\'.")


    def token_clean(self, length=0, punctuation=string.punctuation, numbers=True):

        """
        Strip out non-ascii tokens.
        length: remove tokens of length "length" or less.
        punctuation: string of punctuation to strip out, defaults to string.punctuation
        numbers: strip out numeric tokens.
        """
        def remove_short(tokens):
            return [t for t in tokens if t != '' and len(t) > length] 

        def remove_non_ascii(tokens):
            tokens = [unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8', 'ignore') for t in tokens]
            return [t for t in tokens if t != '' and len(t) > length] 

        def remove_punctuation(tokens, punctuation):            
            regex = re.compile('[%s]' % re.escape(punctuation))
            tokens = [regex.sub('', t) for t in tokens]
            return [t for t in tokens if t != '' and len(t) > length] 
        
        def remove_numbers(tokens):
            tokens = [t.translate(translation_table) for t in tokens if not t.isdigit()] 
            return  [t for t in tokens if t != '' and len(t) > length]

        self.tokens = list(map(remove_non_ascii, self.tokens))
        self.tokens = list(map(remove_short, self.tokens))

        if punctuation is not None:
            self.tokens = list(map(lambda x: remove_punctuation(x,punctuation), self.tokens))

        if numbers:
            translation_table = str.maketrans('', '', string.digits)
            self.tokens = list(map(remove_numbers, self.tokens))


    def stopword_remove(self, items, stopwords="short"):

        """
        Remove stopwords from either tokens (items = "tokens"), lemmas (items = "lemmas")
        or stems (items = "stems")
        
        stopwords: 'long' is longer list of stopwords, 'short' is shorter list
        of stopwords. One can also pass "None" or a customized list of stopwords.
        
        """

        def remove(tokens):
            return [t for t in tokens if t not in self.stopwords]
        

        # initialize the appropriate stopwords list
        if stopwords == 'long':
            self.stopwords = preprocess_data.stp_long
        elif stopwords == 'short':
            self.stopwords = preprocess_data.stp_short
        elif stopwords is None:
            print('No stopwords list initialized. Consider initializing stopwords list\
                to "long" or "short" or passing customized list')
        elif isinstance(stopwords, list):
                # verify that all elements are strings
                if all(isinstance(x, str) for x in stopwords):
                    self.stopwords = stopwords
                else:
                    raise ValueError("All elements of stopwords list must be strings")
        else:
            raise ValueError("Stopwords must be a list of strings")
        
        # apply the stopwords removal to the appropriate items
        if items == 'tokens':
            self.tokens = list(map(remove, self.tokens))
        elif items == 'stems':
            self.stems = list(map(remove, self.stems))
        elif items == 'lemmas':
            self.lemmas = list(map(remove, self.lemmas))
        else:
            raise ValueError("Items must be either \'tokens\', \'lemmas\' or \'stems\'.")


    def stem(self):

        """
        Stem tokens with nltk Snowball Stemmer.
        """

        def s(tokens):
            return [stemmer.stem(t) if "-" not in t else t for t in tokens]

        self.stems = list(map(s, self.tokens))


    def lemmatize(self):

        """
        Lemmatize tokens with spaCy Lemmatizer.
        """
        def lemmatize_sent(sent):
            sent = ' '.join(sent)
            doc = nlp_lemmatizer(sent)
            tokens = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc]
            return tokens
        self.lemmas = list(map(lemmatize_sent, self.tokens))


    def join(self, items):
        """
        Join all elements into a single string separated by white spaces for 
        tokens (items = "tokens"), lemmas (items = "lemmas") or stems (items = "stems")
        """

        def join_elements(elements):
            return ' '.join(elements)

        if items == 'tokens':
            self.tokens = list(map(join_elements, self.tokens))
        elif items == 'stems':
            self.stems = list(map(join_elements, self.stems))
        elif items == 'lemmas':
            self.lemmas = list(map(join_elements, self.lemmas))
        else:
            raise ValueError("Items must be either \'tokens\', \'lemmas\' or \'stems\'.")
    
    def get_term_ranking(self, items, score_type='df'):
        '''
        Produce list of tuple (score, term, ranking) of unique terms 
        score_type: document frequency ('df') default, option for 'tfidf'
        tfidf score computed as tfidf_v = (1+log(tf_v)*(log(N/df_v))
        '''
        if not hasattr(self, 'df_ranking'):
            self.df_ranking = {}

        if not hasattr(self, 'tfidf_ranking'):
            self.tfidf_ranking = {}

        def dummy(doc):
            return doc
        
        if items == 'stems':
            v = self.stems
        elif items == 'tokens':
            v = self.tokens
        elif items == 'bigrams':
            v = self.bigrams
        elif items == 'lemmas':
            v = self.lemmas
        else:
            raise ValueError("Items must be either \'tokens\' , \'bigrams\' , \'lemmas\' or \'stems\'.")

        # apply vectorization
        vectorizer = TfidfVectorizer(use_idf=False, norm=None, tokenizer=dummy, preprocessor=dummy)
        # df_matrix = vectorizer.fit_transform(v).toarray()
        # df_matrix_bool = np.where(df_matrix>0,1,0)
        # scores_df = df_matrix_bool.sum(axis=0)       
        
        df_matrix = vectorizer.fit_transform(v)      
        # get the position of all nonzero elements in the matrix
        nonzero_rows, nonzero_cols = df_matrix.nonzero()
        # create a new matrix
        df_matrix_bool = df_matrix.copy()
        # replace nonzero elements with 1
        df_matrix_bool[nonzero_rows, nonzero_cols] = 1
        scores_df = df_matrix_bool.sum(axis=0)
        scores_df = np.squeeze(np.asarray(scores_df))

        if score_type == 'df':
            sorted_vocab = sorted(vectorizer.vocabulary_.items(),   key=lambda x: x[1])
            sorted_vocab_keys = list(np.array(sorted_vocab)[:,0])
            sorted_scores_df = sorted(set(scores_df), reverse=True)

            rank_dict = {k:val for k,val in zip(sorted_scores_df, list(range(len(sorted_scores_df))))}
            rank_tup = sorted(zip(scores_df, sorted_vocab_keys),  key=lambda x: x[0], reverse=True)
            self.df_ranking[f'{items}'] = [x + (rank_dict[x[0]],) for x in rank_tup]

        elif score_type =='tfidf':
            def tf_idf_compute(t, scores_tf, scores_df):
                return ( 1+np.log(scores_tf[t]) )*np.log( self.N/(scores_df[t]+1) )

            sorted_vocab = sorted(vectorizer.vocabulary_.items(),   key=lambda x: x[1])
            sorted_vocab = list(np.array(sorted_vocab)[:,0])
            scores_tf = df_matrix.sum(axis=0)
            scores_tf = np.squeeze(np.asarray(scores_tf))

            scores_tfidf = [tf_idf_compute(t, scores_tf=scores_tf, scores_df=scores_df) for t in range(len(sorted_vocab))]
            sorted_scores_tfidf = sorted(set(scores_tfidf), reverse=True)

            rank_dict = {k:val for k,val in zip(sorted_scores_tfidf, list(range(len(sorted_scores_tfidf))))}
            rank_tup = sorted(zip(scores_tfidf, sorted_vocab),  key=lambda x: x[0], reverse=True)
            self.tfidf_ranking[f'{items}'] = [x + (rank_dict[x[0]],) for x in rank_tup]
        else: 
            raise ValueError("Score_type must be either \'df\' or \'tfidf\'.")


    def dt_matrix_create(self, items, score_type='df', max_df=1.0, min_df=1, max_tfidf=0, min_tfidf=np.inf, tfidf_norm='l2'):
        '''
        Produce a document term frequency or document tfidf matrix alongside a vocabulary for the specified items. 
        It allows to remove terms with high/low document frequency/tfidf.
        score_type: either 'df' or 'tfidf'
        max_df (min_df): float or int, default=1.0 (default=1). When building the df vocabulary ignore terms that 
        have a document frequency strictly higher (lower) than the given threshold. 
        If float in range [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. 
        max_tfidf (min_tfidf): float or int, default=0 (default=np.inf). When building the tfidf vocabulary 
        ignore terms that have a tfidf across the whole corpus strictly higher (lower) than the given threshold. 
        If float, the parameter represents a tfidf value, integer the ranking (0=highest tfidf ranking, number 
        of unique items=lowest tfidf ranking). 
        tfidf_norm: {‘l1’, ‘l2’, None}, default=’l2’. Normalization technique applied to each term's tfidf score. 
        '''
        
        self.vocabulary = {}
        def dummy(doc):
            return doc

        if items == 'stems':
            v = self.stems
        elif items == 'tokens':
            v = self.tokens
        elif items == 'bigrams':
            v = self.bigrams
        elif items == 'lemmas':
            v = self.lemmas
        else:
            raise ValueError("Items must be either \'tokens\' , \'bigrams\' , \'lemmas\' or \'stems\'.")

        if score_type=='df':
            self.df_matrix = {}
            vectorizer = TfidfVectorizer(use_idf=False, norm=None, max_df=max_df, min_df=min_df, tokenizer=dummy, preprocessor=dummy) 
            self.df_matrix[f'{items}'] = vectorizer.fit_transform(v).toarray()
            self.vocabulary[f'{items}'] = vectorizer.vocabulary_

        elif score_type=='tfidf':
            self.tfidf_matrix = {}
            def remove(tokens):
                tokens = [t for t in tokens if t not in to_remove_low]
                tokens = [t for t in tokens if t not in to_remove_high]
                return tokens

            if not hasattr(self, 'tfidf_ranking'):
                self.get_term_ranking(items, score_type='tfidf')
            ranking = np.array(self.tfidf_ranking[items])

            if type(max_tfidf)==float:
                to_remove_high = set([t[1] for t in ranking if float(t[0]) > max_tfidf])
            else:
                to_remove_high = set([t[1] for t in ranking if float(t[2]) < max_tfidf])

            if type(min_tfidf)==float:
                to_remove_low = set([t[1] for t in ranking if float(t[0]) < min_tfidf])
            else:
                to_remove_low = set([t[1] for t in ranking if float(t[2]) > min_tfidf])

            v = list(map(remove, v))
            vectorizer = TfidfVectorizer(use_idf=True, norm=tfidf_norm, tokenizer=dummy, preprocessor=dummy) 
            self.tfidf_matrix[f'{items}'] = vectorizer.fit_transform(v).toarray()
            self.vocabulary[f'{items}'] = vectorizer.vocabulary_

        else:
            raise ValueError("score_type must be either \'df\' or \'tfidf\'.")

        if items == 'stems':
            self.stems = [[t for t in doc if t in self.vocabulary[f'{items}']] for doc in self.stems]
        elif items == 'tokens':
            self.tokens = [[t for t in doc if t in self.vocabulary[f'{items}']] for doc in self.tokens]
        elif items == 'bigrams':
            self.bigrams = [[t for t in doc if t in self.vocabulary[f'{items}']] for doc in self.bigrams]
        elif items == 'lemmas':
            self.lemmas = [[t for t in doc if t in self.vocabulary[f'{items}']] for doc in self.lemmas]


    def rank_remove(self, items, score_type, min_cutoff_score=-np.inf, max_cutoff_score=np.inf):
        """
        remove items based on the df or tfidf (specified in "score_type") 
        score being less than the cutoff score provided. 
        The get_term_ranking() function needs to be called before.
        """

        # verify that get_term_ranking() has already been called and that
        # a valid item has been provided
        if score_type=="df":
            try:
                self.df_ranking[items]
            except (AttributeError, KeyError):
                print("Please run the get_term_ranking() function first with the appropriate arguments")
                
        elif score_type=="tfidf":
            try:
                self.tfidf_ranking[items]
            except (AttributeError, KeyError):
                print("Please run the get_term_ranking() function first with the appropriate arguments")
                
        else:
            raise ValueError("Score type must be either \'df\' (document frequency) or \'tfidf\' (term frequency inverse document frequency).")

        # auxiliary function to easily remove undesired tokens
        def remove(tokens, to_remove):
            if to_remove:
                return [t for t in tokens if t not in to_remove]
            else:
                return tokens

        # consolidate list of tokens to be removed
        if score_type == "df":
            to_remove_low = set([t[1] for t in self.df_ranking["stems"] if t[0] <= min_cutoff_score])
            to_remove_high = set([t[1] for t in self.df_ranking["stems"] if t[0] > max_cutoff_score])

        elif score_type == "tfidf":
            to_remove_low = set([t[1] for t in self.tfidf_ranking["stems"] if t[0] <= min_cutoff_score])
            to_remove_high = set([t[1] for t in self.tfidf_ranking["stems"] if t[0] > max_cutoff_score])

        # remove tokens from the appropriate place
        if items == 'tokens':
            #self.tokens = list(map(remove, self.tokens))
            self.tokens = [remove(d, to_remove_low) for d in self.tokens]
            self.tokens = [remove(d, to_remove_high) for d in self.tokens]

        elif items == 'lemmas':
            #self.lemmas = list(map(remove, self.lemmas))
            self.lemmas = [remove(d, to_remove_low) for d in self.lemmas]
            self.lemmas = [remove(d, to_remove_high) for d in self.lemmas]

        elif items == 'stems':
            #self.stems = list(map(remove, self.stems))
            self.stems = [remove(d, to_remove_low) for d in self.stems]
            self.stems = [remove(d, to_remove_high) for d in self.stems]
        
        elif items == "bigrams":
            #self.bigrams = list(map(remove, self.bigrams))
            self.bigrams = [remove(d, to_remove_low) for d in self.bigrams]
            self.bigrams = [remove(d, to_remove_high) for d in self.bigrams]
            
            
# %%
