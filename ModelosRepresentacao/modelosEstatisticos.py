"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd 

class ModelosEstatisticos:

    def __init__(self):
        pass
        
    def bow(self, dados):
        bow_modelo = CountVectorizer()
        bow_transformacao = bow_modelo.fit_transform(dados)
        return bow_transformacao

    def tfidf(self, dados):
        tfidf_modelo = TfidfVectorizer(max_df=0.95, min_df=2)
        tfidf_transformacao = tfidf_modelo.fit_transform(dados)   
        return tfidf_transformacao

    def pca(self):
        pass
