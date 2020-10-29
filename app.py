from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import tweepy
from tweepy import OAuthHandler

tfid = pickle.load(open("tfid1.pickle", "rb"))
classifier = pickle.load(open("App.pickle", "rb"))

app = Flask(__name__)


@app.route('/result')
@app.route('/home')
@app.route('/')
def home():
	return render_template('home.html')


# This method is used to determine the Sentiments of tweets and find the emotions present  in it
@app.route('/predict',methods=['POST'])
def predict():
  class TwitterClient(object): 
    def __init__(self): 

        try: 
            # create OAuthHandler object
            consumer_key = 'oatBv3GXUBgzOvovmyCqISn7g'
            consumer_secret = 'Z5OuRSGN4zH64luK9P5o124vboT9dm9BxlXOZPF0FzeoY48dZQ'
            access_token_key = '165322630-d2GYBMk5R1VApKjntW8hcBqDd0Ol4Gfq2JHuVGPK'
            access_token_secret = 'mLC5JPrActYNE94GaA3GIogCsLbQGdk3RN5mKNP3yPnpB'
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token_key, access_token_secret)
            self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
            
        except tweepy.TweepError as e:
            print(f"Error: Tweeter Authentication Failed - \n{str(e)}")

    def get_tweets(self, query, maxTweets = 1000):

        tweets = [] 
        sinceId = None
        max_id = -1
        tweetCount = 0
        tweetsPerQry = 100

        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = self.api.search(q=query, count=tweetsPerQry,tweet_mode='extended',lang = "en")
                    else:
                        new_tweets = self.api.search(q=query, count=tweetsPerQry,
                                                since_id=sinceId,tweet_mode='extended',lang = "en")
                else:
                    if (not sinceId):
                        new_tweets = self.api.search(q=query, count=tweetsPerQry,
                                                max_id=str(max_id - 1),tweet_mode='extended',lang = "en")
                    else:
                        new_tweets = self.api.search(q=query, count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId,tweet_mode='extended',lang = "en")
                if not new_tweets:
                    print("No more tweets found")
                    break

                for tweet in new_tweets:
                    parsed_tweet = {} 
                    parsed_tweet['tweets'] = tweet.full_text 

                    # appending parsed tweet to tweets list 
                    if tweet.retweet_count > 0: 
                        # if tweet has retweets, ensure that it is appended only once 
                        if parsed_tweet not in tweets: 
                            tweets.append(parsed_tweet) 
                    else: 
                        tweets.append(parsed_tweet) 
                        
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:
                # Just exit if any error
                print("Tweepy error : " + str(e))
                break
        
        return pd.DataFrame(tweets)
  def remove_pattern(input_txt, pattern):                                           #function to remove pattern 
      r = re.findall(pattern, input_txt)
      for i in r:
          input_txt = re.sub(i, '', input_txt)        
      return input_txt

  def clean_tweets(lst):
      lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")                         # removing RT @x from tweets:
      lst = np.vectorize(remove_pattern)(lst, "@[\w]*")                             # removing  @xxx from tweets 
      lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")            # reremoving URL links http://xxx
      return lst

  def calculateSentimentsFromTweet(tweets):
      tweets_list = []
      result_list = []
      for i, tweet in enumerate(tweets['clean']):
          op = classifier.predict(tfid.transform([tweet]).toarray())
          if op == [0]:
              tweets_list.append(tweets.tweets[i])
              result_list.append('Negative')
          if op == [1]:
              tweets_list.append(tweets.tweets[i])
              result_list.append('Neutral')
          if op == [2]:
              tweets_list.append(tweets.tweets[i])
              result_list.append('Positive')
      output_dict = dict(zip(tweets_list, result_list))
      return  output_dict,result_list

  def getSentimentCounts(sentiments):
      Neutral_count = sentiments.count('Neutral')
      Negative_count = sentiments.count('Negative')
      Postive_count = sentiments.count('Positive')
      return Neutral_count,Negative_count,Postive_count

  def getEmotionsListFromTweet(sentence):
    emotion_list = []
    sentence = sentence.split(' ')
    with open('emotions.txt','r') as file:
      for line in file:
        clear_line = line.replace("\n", '').replace(",",'').replace("'",'').strip()
        word, emotion = clear_line.split(':')
        if word in sentence:
          emotion_list.append(emotion)
      return emotion_list

  def computeAttributes(emotions):

      happy = emotions.count(' happy')
      sad = emotions.count(' sad')
      anger = emotions.count(' angry')
      love = emotions.count(' loved')
      powerless = emotions.count(' powerless')
      suprise = emotions.count(' surprise')
      fearless = emotions.count(' fearless')
      cheated = emotions.count(' cheated')
      attracted = emotions.count(' attracted')
      singled_out = emotions.count(' singled out')
      anxious = emotions.count(' anxious')
      return happy,sad,anger,love,powerless,suprise,fearless,cheated,attracted,singled_out,anxious



  if request.method == 'POST':
    comment = request.form['Tweet']
    twitter_client = TwitterClient()
    tweet_list = twitter_client.get_tweets(comment, maxTweets=100)
    tweet_list['clean']=clean_tweets(tweet_list['tweets'])
    tweet_list['clean']=tweet_list['clean'].str.replace("[^a-zA-Z ]", " ")
    result_dict,sentiments = calculateSentimentsFromTweet(tweet_list)
    neutral,negative,postive=getSentimentCounts(sentiments)
    emo=getEmotionsListFromTweet(tweet_list['clean'].sum())
    joy,sad,anger,love,powerless,suprise,fearless,cheated,attracted,singledout,anxious = computeAttributes(emo)

    return render_template('result.html',outputs = result_dict,NU=neutral,N=negative,P=postive,happy=joy,sad=sad,angry=anger,loved=love,powerless=powerless,surprise=suprise,fearless=fearless,cheated=cheated,attracted=attracted,singledout=singledout,anxious=anxious)



if __name__ == '__main__':
	app.run(host='localhost', debug ="True")
