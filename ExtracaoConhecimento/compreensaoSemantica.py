
"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd 

class CompreensaoSemantica:

    def __init__(self):
        pass

    def treinamento_intencoes(self, algorithm="labse", sentences, intents):
        ## One-hot encoding of target classes
        intent_classes = list(set(intents))
        y = PreProcessamento("").onehot_encoder(intents, intent_classes)

        if algorithm == "labse":
            model, X_test, y_test = self.labse_model(sentences, y)
        elif algorithm == "bilstm":
            model, X_test, y_test = self.bilstm_model(sentences, y)
        else:
            print("O valor do parâmetro 'algorithm' informado não é válido. Escolha entre 'labse' e 'bilstm'.")
            return

        return model, X_test, y_test, intent_classes


    def bilstm_model(self, sentences, y, n_classes):
        max_seq_len = 280 # max tweet length
        
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Treinar o modelo
        model = LogisticRegressionCV(cv=5, random_state=27).fit(X_train, y_train)
        print("Acurracy: ", model.score(X_test, y_test))
        return model, X_test, y_test


    def predicao_intencoes(self, model, sentences):
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

    
    def extracao_entidades(self):
        pass
