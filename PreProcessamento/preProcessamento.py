import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import spacy 
import re

class PreProcessamento:

    def __init__(self, fonte):
        self.fonte = fonte
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