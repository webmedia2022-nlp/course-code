from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import nltk
from tqdm import tqdm
import pandas as pd 
import keras.utils as np_utils
import bert
import spacy 
import numpy as np
import re

class Preprocessing:

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')

    
    def nlp_pipeline(self, text):

        doc = self.nlp(text)

        tokens, pos_tags, lemmas = [], [], []

        for token in doc:

            if token.text.strip():
                tokens.append(token.text.lower())
                pos_tags.append(token.pos_)
                lemmas.append(token.lemma_.lower())
            else:
                pass

        return tokens, pos_tags, lemmas

    
    def normalization(self, text):
        text = text.lower()
        return text

    
    def tokenization(self, text):
        tokens = word_tokenize(text)
        return tokens
    
    def stemming(self, tokens):
        stem = self.stemmer.stem
        stems = [stem(t) for t in tokens]
        return stems

    
    def lemmatization(self, tokens):
        lemma = self.lemmatizer.lemmatize
        lemmas = [lemma(t) for t in tokens]
        return lemmas

    def pos_tagging(self, tokens):
        pos_tags = nltk.pos_tag(tokens)
        return pos_tags 
    
    def clean_regex(self, text, pattern, value=" "):
        text = re.sub(pattern, " {value} ".format(value=value), text)
        return text.strip()


    def load_tokenizer(self):
        fname = "bert_vocab.txt"
        tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file=fname)
        return tokenizer


    def padding(self, ids, max_seq_len):
        ids = ids + [0] * (max_seq_len - len(ids))
        return np.array(ids)


    def encoding_sentences(self, sentences, max_seq_len):
        X = []
        tokenizer = self.load_tokenizer()
        for text in sentences:
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids)>max_seq_len:
                token_ids = token_ids[:max_seq_len-1]+[token_ids[-1]]
                tokens = tokens[:max_seq_len-1]+[tokens[-1]]
            X.append(self.padding(token_ids, max_seq_len))

        return np.asarray(X)


    def target_encoder(self, y, classes):
        array = []
        for label in tqdm(y):
            array.append(classes.index(label))
        return np.asarray(array)


    def onehot_encoder(self, y, classes):
        array = []
        for label in tqdm(y):
            array.append(classes.index(label))

        # convert class labels to one-hot encoding
        onehot = np_utils.to_categorical(array, num_classes=len(classes))
        encoded_data = np.array(onehot)
        return encoded_data

    def spacy_simple_preprocess(self, text):

        # Usa o Spacy para lematizar o texto e remover stop words
        spacy_doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in spacy_doc if token.lemma_.lower() and not token.is_stop]
        return tokens

    def tokens_to_text(self, tokens):
        return ' '.join([token.strip() for token in tokens if len(token.strip()) > 0])