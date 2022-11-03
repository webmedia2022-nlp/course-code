"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Oct 7th , 2022.

"""
from sentence_transformers import SentenceTransformer
from laserembeddings import Laser
import tensorflow_hub as hub
import tensorflow_text
import torch

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
import pandas as pd
import os
import nltk



from pathlib import Path




class SentenceEmbeddings:

    def __init__(self):
        pass


    def validate_language(self, lang):
        try:
            Lang(lang)
        except InvalidLanguageValue as e:
            print(e.msg)
            return False
        return True


    def infersent(self, sentences):
        fname = "data/infersent/"
        fdir = Path(fname)
        if not fdir.exists():
            Path(fname).mkdir(parents=True, exist_ok=True)
            Path(fname+"GloVe").mkdir(parents=True, exist_ok=True)
            Path(fname+"fastText").mkdir(parents=True, exist_ok=True)
            Path(fname+"encoder").mkdir(parents=True, exist_ok=True)
            os.system(f"curl -Lo {fname}GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip")
            os.system(f"curl -Lo {fname}fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
            os.system(f"unzip {fname}fastText/crawl-300d-2M.vec.zip -d {fname}fastText/")
            os.system(f"curl -Lo {fname}encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl")
            os.system(f"curl -Lo {fname}encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl")
            os.system(f"curl -Lo {fname}models.py https://raw.githubusercontent.com/facebookresearch/InferSent/main/models.py")
        else:
            pass
        from data.infersent.models import InferSent
        
        
        # Load pretrained model
        V = 2
        MODEL_PATH = f'{fname}encoder/infersent{V}.pkl'
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        infersent = InferSent(params_model)

        # Keep it on CPU or put it on GPU
        use_cuda = False
        infersent = infersent.cuda() if use_cuda else infersent

        infersent.load_state_dict(torch.load(MODEL_PATH))

        # Set word vector path for the model
        W2V_PATH = f'{fname}fastText/crawl-300d-2M.vec'
        infersent.set_w2v_path(W2V_PATH)

        # Build the vocabulary of word vectors (i.e keep only those needed)
        infersent.build_vocab(sentences, tokenize=True)

        # Encode your sentences (list of n sentences)
        embeddings = infersent.encode(sentences, tokenize=True)
        return embeddings

    
    def use(self, sentences):
        # Load pre-trained universal sentence encoder model
        use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        embeddings = use(sentences)
        return embeddings.numpy()


    def muse(self, sentences):
        use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        embeddings = use(sentences)
        return embeddings.numpy()


    def sbert(self, sentences):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        return embeddings


    def laser(self, sentences, lang = "en"):
        laser = Laser("data/93langs.fcodes", "data/93langs.fvocab", "data/bilstm.93langs.2018-12-26.pt")
        # lang is only used for tokenization
        if (isinstance(lang, list)):
            if len(sentences) != len(lang):
                raise ValueError('Você deve indicar o código de idioma (ISO 639-1) para cada sentença da lista. Caso todas as senteças sejam do mesmo idioma, basta informar o código de idioma como sendo uma string. Por padrão, temos lang="en".')
            
            for l in lang:
                if not self.validate_language(l):
                    return
        else:
            if not self.validate_language(lang):
                return

        embeddings = laser.embed_sentences(sentences, lang=lang)
        
        return embeddings


    def labse(self, sentences):
        model = SentenceTransformer('sentence-transformers/LaBSE')
        embeddings = model.encode(sentences)
        return embeddings
        

