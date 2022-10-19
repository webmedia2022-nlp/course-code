
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
        
    def word2vec(self, dados):
        return Word2Vec(dados, min_count=1)

    def glove(self, dados):
        pass

    def fasttext(self,dados):
        return FastText(dados, min_count=1)
