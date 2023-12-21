## Import required libraries

## warnings
import warnings
warnings.filterwarnings("ignore")

## for data
import numpy as np
import pandas as pd
import argparse
import random
from PIL import Image

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import nltk
import re
import ftfy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def parse_args():

    parser = argparse.ArgumentParser(description="Process input arguments")

    parser.add_argument(
	    "--dep_data",
        type=str,
        dest="depressive_tweets",
        required = True   
        )
    parser.add_argument(
	    "--rand_data",
        type=str,
        dest="random_tweets",
        required = True
        )
    parser.add_argument(
        "--preprocessed_data",
        type=str,
        dest="preprocessed_data",
        help="data folder mounting point",
        )
    return parser.parse_args()

args = parse_args()

## Read the dataset
depressive_tweets_df = pd.read_csv(args.depressive_tweets)
random_tweets_df = pd.read_csv(args.random_tweets)

## Slicing the random tweets to have sentiment == 1
new_rand_df = random_tweets_df[random_tweets_df.Sentiment == 1]
new_rand_df.reset_index(inplace=True)

## Data Cleaning

print(depressive_tweets_df.shape)
print(new_rand_df.shape)
print(new_rand_df.head())

## Check the data type of each column
depressive_tweets_df.dtypes.to_frame().rename(columns={0:'data_type'})
## Check the data type of each column
new_rand_df.dtypes.to_frame().rename(columns={0:'data_type'})

## Drop unnecessary columns
depressive_tweets_df.drop(columns=['Unnamed: 0'], inplace=True)
new_rand_df.drop(columns=['ï»¿ItemID', 'index','Sentiment', 'SentimentSource'], inplace=True)

## Finding unique values in each column
for col in depressive_tweets_df:
    print("There are ", len(depressive_tweets_df[col].unique()), "unique values in ", col)
'''
## Finding unique values in each column
for col in new_rand_df:
    print("There are ", len(new_rand_df[col].unique()), "unique values in ", col)
'''
## drop duplicate values in tweet.id
depressive_tweets_df.drop_duplicates(subset=['tweet.id'], inplace=True)
depressive_tweets_df.reset_index(inplace=True)
print (depressive_tweets_df.shape)

## Find the number of Null values in each columns
depressive_tweets_df.isnull().sum().to_frame().rename(columns={0:'Null values'})
## Find the number of Null values in each columns
new_rand_df.isnull().sum().to_frame().rename(columns={0:'Null values'})

## Drop all the columns except index, tweet.id and text
new_dep_df = depressive_tweets_df[['text']]
## Add label to both datasets (0 is non-depressive and 1 is depressive)
new_dep_df['label'] = pd.Series([1 for x in range(len(new_dep_df.index))])
#new_rand_df['label'] = pd.Series([0 for x in range(len(new_rand_df.index))])
print (new_dep_df)

## Change the column name to be aligned with depressive dataset
new_rand_df.rename(columns={'SentimentText': 'text'}, inplace=True)
print (new_rand_df)

## Combine two dataframes together
df_all = pd.concat([new_dep_df, new_rand_df], ignore_index=True)
df_all = new_dep_df
print (df_all)

# Expand Contraction
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


## Function to perform stepwise cleaning process
def tweets_cleaner(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = tweet.lower()  # lowercase

        # if url links then don't append to avoid news articles
        # also check tweet length, save those > 5
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 5:
            # remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())

            # fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)

            # expand contraction
            tweet = expandContractions(tweet)

            # remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            # stop words and lemmatization
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(tweet)

            lemmatizer = WordNetLemmatizer()
            filtered_sentence = [lemmatizer.lemmatize(word) for word in word_tokens if not word in stop_words]
            # back to string from list
            tweet = ' '.join(filtered_sentence)  # join words with a space in between them

            cleaned_tweets.append(tweet)

    return cleaned_tweets

tweets_arr = [x for x in df_all['text']]
corpus = tweets_cleaner(tweets_arr)
print (corpus[:10])

## Adding clean tweets as a new column
df_all['clean_text'] = corpus

# replace field that's entirely space (or empty) with NaN
df_all.replace(r'^\s*$', np.nan, regex=True, inplace=True)

## Deleting the rows with nan
df_all.dropna(subset=['clean_text'], inplace=True)

## Double_check for nan
print (df_all[df_all['clean_text'].isnull()])

## Save cleaned_dataset
df_all.to_csv('preprocessed_data.csv',
              sep='\t', encoding='utf-8',index=False)






