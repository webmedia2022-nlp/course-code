
"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""
from Preprocessing import Preprocessing
from TextRepresentation import SentenceEmbeddings


from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.layers.wrappers import Bidirectional

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.0)

import pandas as pd 
import numpy as np
import spacy

class SemanticComprehension:

    def __init__(self):
        pass

    def training_intents(self, algorithm, sentences, intents):
        intent_classes = list(set(intents))
        y = Preprocessing().target_encoder(intents, intent_classes)
        if algorithm == "labse":
            y = Preprocessing().target_encoder(intents, intent_classes)
            model, X_test, y_test = self.labse_model(sentences, y)
        elif algorithm == "bilstm" or algorithm == "rnn":
            ## One-hot encoding of target classes
            onehot_y = Preprocessing().onehot_encoder(intents, intent_classes)
            model, X_test, y_test = self.bilstm_model(sentences, y, onehot_y, len(intent_classes))
        else:
            print("O valor do parâmetro 'algorithm' informado não é válido. Escolha entre 'labse' e 'bilstm'.")
            return

        return model, X_test, y_test, intent_classes


    def bilstm_model(self, sentences, y, onehot_y, n_classes):
        max_seq_len = 280 # max tweet length
        vocab_size = 30522
        
        ## Bidirectional LSTM layer
        inputs = Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, input_length=max_seq_len, mask_zero=True)(inputs)
        embedding_layer = Dropout(0.5)(embedding_layer)
        biRNN,fwd_h, fwd_c, bwd_h, bwd_c = Bidirectional(LSTM(units=300, return_state=True, return_sequences=True))(embedding_layer)
        biRNN = Dropout(0.5)(biRNN)

        ## LSTM layer
        lstm = LSTM(units=300)(biRNN)
        
        softmax = Dense(units=n_classes, activation='softmax')(lstm)
        model = Model(inputs=inputs, outputs=softmax, name='intent_model')
        model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])
        

        X = Preprocessing().encoding_sentences(sentences, max_seq_len)
        print(X.shape)

        ## Separa os dados para o treinamento e teste
        skf = StratifiedKFold(n_splits=2)


        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = onehot_y[train_index], onehot_y[test_index]



        history = model.fit(x=X_train,
                            y=y_train,
                            batch_size=32,
                            epochs= 20,
                            validation_data=(X_test, y_test)
                            )

        ## Evaluate model
        score = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy:", score[1])
        return model, X_test, y_test


    def labse_model(self, sentences, y):
        ## Obter os sentence embeddings utilizando o LaBSE
        X = SentenceEmbeddings().labse(sentences)

        ## Separa os dados para o treinamento e teste
        skf = StratifiedKFold(n_splits=2)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


        # Treinar o modelo
        model = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
        print("Acurracy: ", model.score(X_test, y_test))
        return model, X_test, y_test


    def predict_intents(self, model, sentences):
        # Predição de intents para os tweets coletados
        X = SentenceEmbeddings().labse(sentences)
        X.shape

        y_hat = model.predict(X)
        conf_dist = model.decision_function(X)
        confidence = [round(conf_dist[i][y_hat[i]],3) for i in range(y_hat.shape[0])]

        predicted_intents = []
        for i in range(len(y_hat)):
            predicted_intents.append({
            "text": sentences[i],
            "intent": intent_classes[y_hat[i]],
            "confidence":confidence[i]
            })

        intents = pd.DataFrame(predicted_intents)
        return intents

    
    def plot_confusion_matrix(self, y_test, y_hat, classes, fname):
        cm = confusion_matrix(y_test, y_hat, normalize='true')
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        
        fig, ax = plt.subplots(figsize=(20, 15))
        hmap = sns.heatmap(df_cm, ax=ax, annot=True, fmt=".2")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        return


    def ner(self, sentence):
        entities = []
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sentence)
        entity = {}
        for i, ent in enumerate(doc.ents):
            entity[i] = {
                        "value":ent.text,
                        "entity":ent.label_,
                        "start":ent.start_char,
                        "end":ent.end_char
                    }
            entities.append(entity)

        return entities