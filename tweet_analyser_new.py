from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import twitter_credentials as credentials
from textblob import TextBlob

import pandas as pd
import re
import numpy as np

class tweetListener(StreamListener):
    """
    Listener that gives out tweets received by stdout.
    """
    
    def __init__(self, file_of_tweets):
        self.file_of_tweets = file_of_tweets
    
    def on_data(self, data):
        try:
            with open(self.file_of_tweets, 'a') as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"Error with on_data method: {e}")
        return True
    
    def on_error(self, error):
        if error == 420:
            #error 420 is given out by twitter when it feels that you are abusing their data
            #happens when rate limits occur
            print("Not a good time to access tweets. Killing the connection.")
            return False
        print(error)
        
    #end of class tweetListener
    
class twitterAuthenticator():
    """
    Uses OAuthHandler to authenticate keys and tokens from Twitter API
    """
    def authenticate(self):
        #method to obtain keys or tokens
        auth = OAuthHandler(credentials.consumer_key, credentials.consumer_key_secret)
        auth.set_access_token(credentials.access_token, credentials.access_token_secret)
        return auth
    
    #end of class twitterAuthenticator

class twitterStreamer():
    """
    This class streams and processes live tweets.
    """
    def __init__(self):
        self.twitterAuthenticatorObj = twitterAuthenticator()
        
    def stream_tweets(self, file_of_tweets, list_of_keywords):
        # method to handle twitter authentication and API
         listener = tweetListener(file_of_tweets)
         auth = self.twitterAuthenticatorObj.authenticate()
    
         #streaming
         stream = Stream(auth, listener)
         stream.filter(track=list_of_keywords)
     
    #end of class twitterStreamer   

class twitterClient():
    """
    Gets tweets from specific timelines
    """
    def __init__(self,twitter_user=None):
        self.auth = twitterAuthenticator().authenticate()
        self.client = API(self.auth)
        self.twitter_user = twitter_user
        
    def get_twitter_client(self):
        return self.client
    
    def get_tweets_from_user(self,num_of_tweets):
        #gets tweets from given username
        list_of_tweets = []
        for tweet in Cursor(self.client.user_timeline, id=self.twitter_user).items(num_of_tweets):
            list_of_tweets.append(tweet)
        return list_of_tweets
    
    def get_friendlist_from_client(self,num_of_friends):
        list_of_friends=[]
        for friend in Cursor(self.client.friends, id=self.twitter_user).items(num_of_friends):
            list_of_friends.append(friend)
        return list_of_friends

    def get_tweets_from_home_timeline(self, num_of_tweets):
        list_of_home_tweets=[]
        for tweet in Cursor(self.client.home_timeline,id=self.twitter_user).items(num_of_tweets):
            list_of_home_tweets.append(tweet)
        return list_of_home_tweets
        
    def get_keyworded_tweets(self, num_of_tweets, keyword):
        keyword_tweets = []
        for tweet in Cursor(self.client.search,q=[keyword]).items(num_of_tweets):
            keyword_tweets.append(tweet)
        return keyword_tweets
    #end of twitterClient

class tweetAnalyzer():
    """
    methods to analyse tweets
    """        
    def tweets_to_dataframe(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns = ['Tweets'])
        #df['Tweet ID'] = np.array([tweet.id for tweet in tweets])
        #df['Word Count'] = np.array([len(tweet.text) for tweet in tweets])
        #df['Date'] = np.array([tweet.created_at for tweet in tweets])
        #df['Likes'] = np.array([tweet.favourites_count for tweet in tweets])
        #df['Retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        #df['Source'] = np.array([tweet.source for tweet in tweets])
        return df
        
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        
        if analysis.sentiment.polarity > 0:
            return 2
        elif analysis.sentiment.polarity == 0:
            return 1
        else:
            return 0
    #end of tweetAnalyzer

if __name__ == "__main__":
    twitterClientObj = twitterClient()
    tweetAnalyzerObj = tweetAnalyzer()
    
    client_api = twitterClientObj.get_twitter_client()

    keyword = input("Enter the keyword for which the tweets are to be extracted:")
    
    #ip = input("Enter Twitter ID of person whose tweets are to be analysed:")
    #count_of_ip_tweets = int(input("Enter the number of tweets to be analysed:"))
    #tweets = client_api.user_timeline(screen_name=ip, count=count_of_ip_tweets)
    keyworded_tweets=twitterClientObj.get_keyworded_tweets(50,keyword)
    #print(tweets)
    tweet_df = tweetAnalyzerObj.tweets_to_dataframe(keyworded_tweets)
    #tweet_df['Sentiment'] = np.array([tweetAnalyzerObj.analyze_sentiment(tweet) for tweet in tweet_df['Tweets']])
    
    print(tweet_df)
    header = ["Tweets"]
    tweet_df.to_csv(keyword+"_tweets.csv", columns = header)
    
    # print(f"The average sentiment of the tweets is {round(np.mean(tweet_df['Sentiment']),4)}")

    
