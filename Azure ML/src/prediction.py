import warnings
warnings.filterwarnings("ignore")

import preprocess as p
import numpy as np
import pandas as pd

## for saving and loading model
import pickle

## for word embedding with Spacy
import spacy
import en_core_web_lg
###############################################################
test_tweet = "I am excited about visiting you"
print("Tweet: ", test_tweet)
## Call tweets_cleaner function to clean the tweet
print("cleaning...")
clean_tweet = []
clean_tweet.append(p.tweets_cleaner(test_tweet))

print("Clean tweet:", clean_tweet)

## load English model of Spacy
nlp = en_core_web_lg.load()
## word-embedding
test = pd.np.array([pd.np.array([token.vector for token in nlp(s)]).mean(axis=0) * pd.np.ones((300)) \
                    for s in clean_tweet])

SVM = "D:\\Projects\\NCI_Fall_2023\\MLHops\\Depression_Tweets\\models\\model_svm1.pkl"
with open(SVM, 'rb') as file:
    clf = pickle.load(file)
    print("model==> ", clf)

## prediction
labels_pred = clf.predict(test)
result = labels_pred[0]

if result == 1:
    print("Your tweet seems to be depressive")
else:
    print("Your tweet is non-depressive")
