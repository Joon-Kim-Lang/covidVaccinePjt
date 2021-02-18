from django.apps import AppConfig
import threading
import time
import tweepy
import pandas as pd
import numpy as np
import datetime

test = '무'

def auto_crawling():
    flag = True
    while(True):
        if flag == True:
            vcName = "moderna"
            fname = "moderna_tweets_0217.csv"
        else:
            #화이자로 바꿔야할 부분
            vcName = "moderna"
            fname = "moderna_tweets_0217.csv"
        time.sleep(20)
        access_token = "1359155396695502849-9feqBFFBtVen65QaiPrrtbUNu3lYDa"
        access_token_secret = "MQTYKlCG3cFiZ20dMbwxw7vjQ3foOo1xNpAZXuvKPBAN2"
        consumer_key = "6Em7KBfky16lnxG0uFgD2l2pX" #api key
        consumer_secret ="em2hzJHPD0rX7rgWpISw8ZITRP8Y9AycSGCx4jh5t75PV8AHia" #api secret key

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth,wait_on_rate_limit=True)

        result = []
        max_tweets = 5

        searched_tweets = []
        last_id = -1
        while len(searched_tweets) < max_tweets:
            count = max_tweets - len(searched_tweets)
            print("we have crawled ",len(searched_tweets)," tweets.")
            try:
                new_tweets = api.search(q=vcName, count=count, max_id=str(last_id - 1),result_type="recent", lang="en")
                if not new_tweets:
                    print("no tweets!")
                    break
                print("we have tweets!")
                for tweet in new_tweets:
                    if "RT" not in tweet.text:
                        searched_tweets.append([tweet.id, tweet.user.name, tweet.user.location, tweet.user.description, tweet.user.created_at,
                                       tweet.user.followers_count,
                                       tweet.user.friends_count,
                                       tweet.user.favourites_count,
                                       tweet.user.verified,
                                       tweet.created_at,
                                       tweet.text,
                                       tweet.entities["hashtags"],
                                       tweet.source,
                                       tweet.retweet_count,
                                       tweet.favorite_count,
                                       tweet.retweeted
                                      ])
                last_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                # depending on TweepError.code, one may want to retry or wait
                # to keep things simple, we will give up on an error
                break

        df = pd.DataFrame(data = searched_tweets, columns = ["id" ,"user_name" ,"user_location", "user_description", "user_created", "user_followers",  "user_friends", "user_favourites","user_verified",  "date",  "text","hashtags", "source", "retweets", "favorites", "is_retweet"])
        df["hashtags"] = df["hashtags"].apply(lambda x: handleHT(x))
        df.drop_duplicates(subset=["user_name","text"],inplace=True,keep="last")

        global test
        test = df.copy()
        # df.to_csv(fname,mode='a')
        flag = not flag


def handleHT(hashtags):
    temp = []
    if len(hashtags)==0:
        hashtags=np.nan
    else:
        for tag in hashtags:
            temp.append(tag["text"])
        hashtags = temp
    return hashtags

#자동 크롤링 쓰레드
class VaccinepjtConfig(AppConfig):
    name = 'vaccinePjt'

    def ready(self):
        _thread = threading.Thread(target=auto_crawling)
        _thread.setDaemon(True)
        _thread.start()

        pass
