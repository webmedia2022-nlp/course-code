from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
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

    def nlp_pipeline(self, texto):

        doc = self.nlp(texto)

        tokens, pos_tags, lemmas = [], [], []

        for token in doc:
            tokens.append(token.text)
            pos_tags.append(token.pos_)
            lemmas.append(token.lemma_)

        return tokens, pos_tags, lemmas

    def normalizacao(self, texto):
        texto_normalizado = texto.lower()
        return texto_normalizado

    def tokenizacao(self, texto):
        tokens = word_tokenize(texto)
        return tokens
    
    def stemmizacao(self, tokens):
        stem = self.stemmer.stem
        stems = [stem(t) for t in tokens]
        return stems

    def lemmatizacao(self, tokens):
        lemma = self.lemmatizer.lemmatize
        lemmas = [lemma(t) for t in tokens]
        return lemmas
    
    def limpeza_regex(self, texto, padrao, valor=" "):

        texto = re.sub(padrao, " {valor} ".format(valor=valor), texto)

        return texto


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