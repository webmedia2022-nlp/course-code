"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA

import pandas as pd 

class StatisticalModels:

    def __init__(self):
        pass
        
    def bow(self, dados):
        bow_modelo = CountVectorizer()
        bow_transformacao = bow_modelo.fit_transform(dados)
        return bow_modelo, bow_transformacao

    def tfidf(self, dados):
        tfidf_modelo = TfidfVectorizer(max_df=0.95, min_df=2)
        tfidf_transformacao = tfidf_modelo.fit_transform(dados)   
        return tfidf_modelo, tfidf_transformacao

    def pca(self, n_components = 2, dados):

        pca_model = PCA(n_components=n_components)
        pca_transformacao = pca_model.fit_transform(dados)

        return pca_model, pca_transformacao
