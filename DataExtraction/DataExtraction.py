
"""

Authors: 
    
    Fabio Rezende (fabiorezende@usp.br) 
    Frances Santos (frances.santos@ic.unicamp.br)
    Jordan Kobellarz (jordan@alunos.utfpr.edu.br)

Updated in Sep 28th , 2022.

"""

from tqdm import tqdm
import requests
import tweepy
import praw
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
                if os.path.exists("data/tweets.json"):
                    with open("data/tweets.json", "r") as tweet_file:
                        tweets += json.load(tweet_file)
                with open("data/tweets.json", "w") as tweet_file:
                    json.dump(tweets, tweet_file, indent=2)
                self.running = False
        time.sleep(0.2)

class DataExtraction:

    def __init__(self):
        pass

    
    def twitter(self, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, BEARER_TOKEN):
        global tweets

        client = tweepy.Client(BEARER_TOKEN, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
        auth = tweepy.OAuth1UserHandler(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth)
        stream = TwitterListener(bearer_token=BEARER_TOKEN)
        stream.add_rules(tweepy.StreamRule('nyc'))
        stream.filter(tweet_fields=["created_at", "entities", "geo", "lang", "public_metrics", "source"])

        with open("data/tweets.json", "r") as tweet_file:
            tweets = json.load(tweet_file)
            data = [{
                #"id": item["id"],
                "created_at": item["created_at"],
                "url": "https://twitter.com/anyuser/status/" + item["id"],
                #"retweet_count": item["public_metrics"]["retweet_count"],
                #"reply_count": item["public_metrics"]["reply_count"],
                #"like_count": item["public_metrics"]["like_count"],
                #"quote_count": item["public_metrics"]["quote_count"],
                #"source": item["source"],
                "score": item["public_metrics"]["like_count"],
                "text": item["text"],
                "length":len(item["text"]),
                "geo": item["geo"]
                } for item in tweets]

        tweets = []
        return pd.DataFrame(data)

    def reddit(self, CLIENT_ID, CLIENT_SECRET, subreddits=[], top_n=10, save_to='data/reddit_posts.csv'):
        
        # create Reddit client
        reddit = praw.Reddit(
            client_id=CLIENT_ID, 
            client_secret=CLIENT_SECRET, 
            user_agent='webmedia'
        )

        # get top_n hottest posts from each subreddit
        posts = []
        for subreddit in tqdm(subreddits):
            for post in reddit.subreddit(subreddit).top('all', limit=top_n):

                posts.append({
                    'created_at': post.created,
                    'url': post.url, 
                    #'title': post.title, 
                    'score': post.score, # number of upvotes
                    #'num_comments': post.num_comments,
                    'text': post.selftext,
                    'length': len(post.selftext)
                })

        # create and persist dataframe
        df_posts = pd.DataFrame(posts)
        df_posts.to_csv(save_to)

        return df_posts

    def facebook(self, API_TOKEN, search_term, start_date, end_date, save_to='data/facebook_posts.csv'):

        # prepare API request
        api_url = 'https://api.crowdtangle.com/posts?token=' + API_TOKEN
        api_url += '&searchTerm=' + search_term
        api_url += '&count=' + str(100) # the API allow a maximun of 100 posts per query
        api_url += '&language=en'
        api_url += '&startDate=' + start_date
        api_url += '&endDate=' + end_date

        response = requests.get(api_url)
        data = response.json()
        
        # transform response
        posts = []
        for post in data['result']['posts']:
            posts.append({
                'created_at': post['date'],
                'url': post['postUrl'],
                #'title': None,
                'score': post['statistics']['actual']['likeCount'], 
                'text': post['message'],
                'length': len(post['message'])
            })

        # create and persist dataframe
        df_posts = pd.DataFrame(posts)
        df_posts.to_csv(save_to)
        
        return df_posts
        