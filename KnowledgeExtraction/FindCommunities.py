
"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

import networkx as nx
from networkx.algorithms.community import label_propagation_communities
import pandas as pd
import numpy as np
import json


class FindCommunities:

    def __init__(self):
        pass


    def create_graph(self, vocab, model):
        ## Creating the Graph object
        G = nx.Graph()

        print("Adicionando nós...")
        G = self.add_graph_nodes(G, vocab)

        print("Adicionando arestas...")
        G = self.add_graph_edges(G, model)

        print("Feito!")

        print("Número de nós: ", G.number_of_nodes())
        print("Número de arestas: ", G.number_of_edges())
        print("Número de componentes conectados: ", nx.number_connected_components(G))

        return G


    def add_graph_nodes(self, G, vocab):
        for i, word in enumerate(vocab):
            G.add_node(i, label=word, community=[])
        return G

    
    def add_graph_edges(self, G, model):
        keys = G.nodes().keys()
        for i in keys:
            wordA = G.nodes[i]['label']
            for j in keys:
                if i != j and not G.has_edge(j,i):
                    wordB = G.nodes[j]['label']
                    sim = model.wv.similarity(wordA, wordB)
                    G.add_edge(i, j, weight=sim)
                else:
                    pass
        return G


    def remove_isolated_nodes(self, G):
        to_remove = []
        for u in G.nodes():
            if G.degree(u) < 1:     
                to_remove.append(u)
        for u in to_remove:
            G.remove_node(u)
        del to_remove[:]
        return G


    def remove_edges(self, G):
        ## Get the minimum and maximum weights
        edges = {}
        for u in G.nodes():
            for v in G.neighbors(u):
                edges[(u,v)] = G[u][v]["weight"]
        vals = np.fromiter(edges.values(), dtype=float)
        wmin = np.amin(vals)
        wmax = np.amax(vals)

        ## Normalize the weights
        edges = {}
        for u in G.nodes():
            for v in G.neighbors(u):
                G[u][v]["norm_weight"] = self.normalize_weight(G[u][v]["weight"], wmax, wmin)
                edges[(u,v)] = G[u][v]["norm_weight"]
        
        ## Compute a suitable threshold
        vals = np.fromiter(edges.values(), dtype=float)
        edges.clear()
        wmin = np.amin(vals)
        wmax = np.amax(vals)
        avg = np.mean(vals)
        std = np.std(vals)
        thresh = avg+2.5*std
        #print("Peso máximo: {} Peso mínimo: {}".format(wmax, wmin))
        #print("Média: {}\nDesvio padrão:{}\nThreshold:{}".format(avg, std, thresh))
        print("Threshold:{}".format(thresh))

        ## Remove lightweight edges, i.e., which have weight lower than the threshold
        to_remove = []
        for u in G.nodes():            
            for v in G.neighbors(u):
                if G[u][v]["norm_weight"] >= thresh:
                    pass
                else:
                    to_remove.append([u,v])
        
        print("Deletando {} arestas ...".format(len(to_remove)))
        for edge in to_remove:
            if G.has_edge(edge[0],edge[1]):
                G.remove_edge(edge[0],edge[1])
        del to_remove[:]
        return G


    def normalize_weight(self, wi, wmax, wmin):
        zi = (wi - wmin) / (wmax - wmin)
        return zi


    def lpa(self, G):
        ## Remove lightweight_edges
        G = self.remove_edges(G)
        G = self.remove_isolated_nodes(G)
        print("Número de nós: ", G.number_of_nodes())
        print("Número de arestas: ", G.number_of_edges())
        print("Número de componentes conectados: ", nx.number_connected_components(G))

        comm = list(label_propagation_communities(G))
        n_comm = len(comm)
        print("Identified %d communities" % n_comm)
        if n_comm > 0:
            for i, members in enumerate(comm):
                print("Community", i)
                print([G.nodes[m]['label'] for m in members])
                for m in members:
                    G.nodes[m]['community'].append(i)
        else:
            return G, n_comm
        return G, n_comm 