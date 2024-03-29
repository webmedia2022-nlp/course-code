
"""

Authors: 

    Fabio Rezende (fabiorezende@usp.br)
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


class Clustering:

    def __init__(self):
        pass

    def kmeans(self, data, k):

        kmeans_model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=1,
            random_state=100
        )

        return kmeans_model.fit(data)

    def kmeans_pca_plot(self, data, clusters, plot_n=1000):
        # Baseado em https://www.kaggle.com/code/jbencina/clustering-documents-with-tfidf-and-kmeans/notebook
        pca = PCA(n_components=2).fit_transform(data.todense())

        # seleciona plot_n pontos para apresentar
        idx = np.random.choice(range(pca.shape[0]), size=plot_n, replace=False)
        clusters = [cm.hsv(k/max(clusters)) for k in clusters[idx]]

        plt.scatter(pca[idx, 0], pca[idx, 1], c=clusters)
        plt.title('PCA Clusters')

    def hierarquical_dendrogram(self, data, linkage_method='ward'):
        dendrogram(linkage(data, metric='euclidean', method=linkage_method))
        plt.title('Dendrograma')
        plt.xlabel('Textos')
        plt.ylabel('Distância Euclidiana')

    def hierarquical(self, data, n_clusters, linkage_method='ward'):
        ac_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage_method)
        return ac_model.fit(data)
