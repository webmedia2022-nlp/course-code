
"""

Authors:

    Fabio Rezende (fabiorezende@usp.br)
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

import pandas as pd
from nrclex import NRCLex
from googleapiclient import discovery


class SentimentAnalysis:

    def __init__(self):
        pass

    def emolex(sentences):
        data = []

        for sentence in sentences:
            nrc_emotion = NRCLex(sentence)
            result = nrc_emotion.affect_frequencies
            data.append({**{'text': sentence}, **result})

        return pd.DataFrame(data)

    def perspective(sentences, PERSPECTIVE_API_KEY,
                    attributes=None, language='en'):

        if type(sentences) is not list:
            raise Exception('sentences parameter must be a list os string sequences.')

        # Cria o cliente do Perspective API via Google Cloud discovery
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PERSPECTIVE_API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False
        )

        # Se nenhum atributo for passado, esses serão os default
        if attributes is None:
            attributes = [
                'TOXICITY',
                'SEVERE_TOXICITY',
                'IDENTITY_ATTACK',
                'INSULT',
                'PROFANITY',
                'THREAT',
            ]

        # prepara o corpo da requisição
        requested_attributes = {}
        for attribute in attributes:
            requested_attributes[attribute] = {}

        data = []
        for sentence in sentences:
            # executa a requisição
            response = client.comments().analyze(body={
                'comment': {'text': sentence},
                'requestedAttributes': requested_attributes,
                'languages': [language]
            }).execute()

            # em caso de resposta vazia
            if 'attributeScores' not in response:
                raise Exception

            # transforma a resposta em um dicionário com os atributos do Perspective API como chaves 
            response_as_dict = {'text': sentence}
            for attribute in attributes:
                response_as_dict[attribute] = response['attributeScores'][attribute]['spanScores'][0]['score']['value']

            data.append(response_as_dict)

        return pd.DataFrame(data)

    """
    def sentiment_analysis(self):
        pass

    def toxicity_analysis(self):
        pass
    """