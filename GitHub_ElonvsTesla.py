import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import re
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import nltk

#fig = plt.figure()
plt.show()

#Getting the Stocks csv file
StocksDF = pd.read_csv('TESLASTOCKS.csv')
StocksDF['Date'] = pd.to_datetime(StocksDF['Date'], dayfirst=True)

#Removing the "%" sign from Percent Change colum
Percent_Change = []
for x in range(len(StocksDF["Percent Change"])):
    PercentChange = (StocksDF["Percent Change"][x])
    PercentChange = PercentChange[:-1]
    PercentChange = int(PercentChange)
    Percent_Change.append(PercentChange)
PercentChangeDict = {"Percent Change":Percent_Change}
StocksDF = StocksDF.drop(columns = "Percent Change")
StocksDF["Percent Change"] = Percent_Change

#Getting the Tweets csv file
TweetsDF = pd.read_csv('TWEETS.csv')
TweetsDF = TweetsDF.rename(columns = {'Timestamp':'Date'})
TweetsDF = TweetsDF.drop(['Emojis','Retweets','UserScreenName', 'UserName','Comments','Likes','Image link'],axis=1)
TweetsDF['Date'] = pd.to_datetime(TweetsDF['Date'], dayfirst=True)

#Merging the two dataframes together
df = pd.merge(StocksDF, TweetsDF, on = "Date")

#Dropping duplicate dates by keeping only the first dataset of that particualar date
df = df.drop_duplicates(subset='Date', keep = "first")
df.reset_index(drop=True, inplace=True)

#Sentiment Analysis

#Cleaning Text
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    return text

df["text"] = df["text"].apply(cleanTxt)

#Sentiment Analysis
def percentage(part,whole):
    return 100 * float(part)/float(whole)

#Assigning Initial Values
positive = 0
negative = 0
neutral = 0
#Creating empty lists
tweet_list1 = []
neutral_list = []
negative_list = []
positive_list = []
pos_increased_value = []
neg_increased_value = []
pos_decreased_value = []
neg_decreased_value = []
pos_max_value = []
neg_max_value = []
pos_min_value = []
neg_min_value = []

#Iterating over the tweets in the dataframe
i = 0
for tweet in df['text']:
    tweet_list1.append(tweet)
    analyzer = SentimentIntensityAnalyzer().polarity_scores(tweet)
    neg = analyzer['neg']
    neu = analyzer['neu']
    pos = analyzer['pos']
    comp = analyzer['compound']

    if neg > pos:
        negative_list.append(tweet) #appending the tweet that satisfies this condition
        negative += 1 #increasing the count by 1
        neg_max_value.append(df['High'][i])
        neg_min_value.append(df['Low'][i])
        if df['Percent Change'][i] > 0:
            neg_increased_value.append(df['Percent Change'][i])
        elif df['Percent Change'][i] <= 0:
            neg_decreased_value.append(df['Percent Change'][i])
    if pos > neg:
        positive_list.append(tweet) #appending the tweet that satisfies this condition
        positive += 1 #increasing the count by 1
        pos_max_value.append(df['High'][i])
        pos_min_value.append(df['Low'][i])
        if df['Percent Change'][i] > 0:
            pos_increased_value.append(df['Percent Change'][i])
        elif df['Percent Change'][i] <= 0:
            pos_decreased_value.append(df['Percent Change'][i])
    elif pos == neg:
        neutral_list.append(tweet) #appending the tweet that satisfies this condition
        neutral += 1 #increasing the count by 1 
    
    i += 1

positive = percentage(positive, len(df)) #percentage is the function defined above
negative = percentage(negative, len(df))
neutral = percentage(neutral, len(df))

#Print the percent tweets with a positive, negative or neutral sentiment
print("Pos:", positive)
print("Neg:", negative)
print("Neu:", neutral)

