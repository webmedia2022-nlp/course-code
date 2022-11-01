
"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

import pandas as pd 
from gensim.models import Word2Vec, FastText

class WordEmbeddings:

    def __init__(self):
        pass
        
    def word2vec(self, sentences):
        model = Word2Vec(sentences, min_count=1)
        return model

    def glove(self, sentences):
        pass

    def fasttext(self,sentences):
        return FastText(sentences, min_count=1)
