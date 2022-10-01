
"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""
#from tweepy.streaming import StreamListener
#from tweepy import OAuthHandler
#from tweepy import Stream
import tweepy
import json
import pandas as pd 
import time
import os

tweets = []

class TwitterListener(tweepy.StreamingClient):
    def on_connect(self):
        self.running = True
        print("Conectado")


    def on_data(self, data):
        global tweets
        data = json.loads(data.decode("utf-8"))
        if data["data"]["lang"] == "en":
            if len(tweets) < 10:
                tweets.append(data["data"])
            else:
                print("Salvando 10 novos tweets no disco ...")
                if os.path.exists("tweets.json"):
                    with open("tweets.json", "r") as tweet_file:
                        tweets += json.load(tweet_file)
                with open("tweets.json", "w") as tweet_file:
                    json.dump(tweets, tweet_file, indent=2)
                self.running = False
        time.sleep(0.2)


class ExtracaoDados:

    def __init__(self, metodo):
        self.metodo = metodo
        pass

    
    def twitter(self, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, BEARER_TOKEN):
        global tweets

        client = tweepy.Client(BEARER_TOKEN, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
        auth = tweepy.OAuth1UserHandler(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth)
        stream = TwitterListener(bearer_token=BEARER_TOKEN)
        stream.add_rules(tweepy.StreamRule('nyc'))
        stream.filter(tweet_fields=["created_at", "entities", "geo", "lang", "public_metrics", "source"])

        with open("tweets.json", "r") as tweet_file:
            tweets = json.load(tweet_file)
            data = [{
                "id": item["id"],
                "created_at": item["created_at"],
                "geo": item["geo"],
                "retweet_count": item["public_metrics"]["retweet_count"],
                "reply_count": item["public_metrics"]["reply_count"],
                "like_count": item["public_metrics"]["like_count"],
                "quote_count": item["public_metrics"]["quote_count"],
                "source": item["source"],
                "text": item["text"]
                } for item in tweets]

        tweets = []
        return pd.DataFrame(data)

    def reddit(self):
        return pd.Series()
