from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sentence_transformers import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from sklearn.cluster import MiniBatchKMeansuma
import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.cm as cm
import torch
import sentence_transformers
import umap
from gensim.models import Word2Vec


def Preprocessing(file,country):
  raw_data = file
  raw_data = raw_data.drop(['id','user_created','source'],axis=1)
  data1 = pd.DataFrame(raw_data[['user_location','user_followers','user_favourites','user_verified','favorites','is_retweet']])
  data2 = pd.DataFrame(raw_data['text'])
  #location
  data1['user_location'] = data1['user_location'].str.lower()
  data_co = data1.copy()
  for x in country['location']:
    for k in range(len(data_co)):
      if x in str(data_co['user_location'][k]):
        data_co['user_location'][k] = 1
  for k in range(len(data_co)):
    if 'london' in str(data_co['user_location'][k]):
      data_co['user_location'][k] = 1
  for k in range(len(data_co)):
    if 'uk' in str(data_co['user_location'][k]):
      data_co['user_location'][k] = 1
  for k in range(len(data_co)):
    if data_co['user_location'][k] != 1  :
      data_co['user_location'][k] = 0
  data1 = data_co
  #dropna
  data1 = data1[data1['is_retweet'].isnull()==False]
  data1 = data1.drop(['is_retweet'],axis=1)
  data1 = data1.reset_index(drop = True)
  #normalization
  data1['user_followers']=data1['user_followers']/max(data1['user_followers'])
  data1['user_favourites']=data1['user_favourites']/max(data1['user_favourites'])
  data1['favorites']=data1['favorites']/max(data1['favorites'])
  return data1, data2

  #Data2 처리를 위한 함수
def preprocess_tweet_data(data,name):
    # Lowering the case of the words in the sentences
    data[name]=data[name].str.lower()
    # Code to remove the Hashtags from the text
    data[name]=data[name].apply(lambda x:re.sub(r'#','',x))
    # Code to remove the links from the text
    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    # Code to remove the Special characters from the text
    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Code to substitute the multiple spaces with single spaces
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Code to remove all the single characters in the text
    data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    # Remove the twitter handlers
    data[name]=data[name].apply(lambda x:re.sub('@[^\s]+','',x))
def rem_stopwords_tokenize(data,name):

    def getting(sen):
        example_sent = sen

        stop_words = set(stopwords.words('english'))

        word_tokens = word_tokenize(example_sent)

        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w.lower())
        return filtered_sentence
    x=[]
    for i in data[name].values:
        x.append(getting(i))
    data[name]=x
# Making a function to lemmatize all the words
lemmatizer = WordNetLemmatizer()
def lemmatize_all(data,name):
    arr=data[name]
    a=[]
    for i in arr:
        b=[]
        for j in i:
            x=lemmatizer.lemmatize(j,pos='a')
            x=lemmatizer.lemmatize(x)
            b.append(x)
        a.append(b)
    data[name]=a
# Function to make it back into a sentence
def make_sentences(data,name):
    data[name]=data[name].apply(lambda x:' '.join([i+' ' for i in x]))
    # Removing double spaces if created
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))

#Function to run all above
def preprocess_all(file,country):
  data1, data2 = Preprocessing(file,country)
  name = 'text'
  preprocess_tweet_data(data2,name)
  rem_stopwords_tokenize(data2,name)
  lemmatize_all(data2,name)
  make_sentences(data2,name)
  return data1, data2
