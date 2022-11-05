
"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""
from sklearn.cluster import MiniBatchKMeans
import pandas as pd 

class Clustering:

    def __init__(self):
        pass

    def kmeans(self, data, number_clusters=3):

        kmeans_model = MiniBatchKMeans(
            n_clusters=number_clusters,
            init="k-means++",
            n_init=1,

        )

        predictions = kmeans_model.fit_predict(data)

        return predictions

    def hierarquico(self):
        pass
