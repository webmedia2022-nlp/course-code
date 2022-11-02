
"""

Authors:

    Fabio Rezende (fabiorezende@usp.br)
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

from sklearn.decomposition import NMF, LatentDirichletAllocation

class TopicModeling:


    def nmf(self, tf_idf_model, no_topics=5):
        nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tf_idf_model)
        return nmf_model 

    def lda(self, tf_idf_model, no_topics=5):
        lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
        return lda_model
