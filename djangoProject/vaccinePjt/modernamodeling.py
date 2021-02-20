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
import vaccinePjt.preprocessing as pp

Moderna_country = ''
Moderna_original = ''

data1_moderna,data2_moderna = '',''
Moderna_score_model = ''
Moderna_model = ''
Moderna_embeddings = ''

Moderna_embeddings_backup = ''
Moderna_original_backup = ''


M_new2 = ''

def forModernaInit():
    global Moderna_country
    global Moderna_original
    Moderna_country = pd.read_csv('Moderna_country.csv')
    Moderna_original = pd.read_csv('moderna_tweets_0217.csv')[:100]
    global data1_moderna
    global data2_moderna
    data1_moderna,data2_moderna = pp.preprocess_all(Moderna_original, Moderna_country)

    global Moderna_score_model
    Moderna_score_model = KMeans(n_clusters=2).fit(data1_moderna)
    Moderna_original['score'] = Moderna_score_model.labels_

    global Moderna_model
    global Moderna_embeddings
    global Moderna_embeddings_backup
    global Moderna_original_backup
    Moderna_model=SentenceTransformer('bert-large-nli-mean-tokens')
    Moderna_embeddings = Moderna_model.encode(data2_moderna['text'])
    Moderna_embeddings_backup = Moderna_embeddings.copy()
    Moderna_original_backup = Moderna_original.copy()
    print("Server successfull loaded")

def Moderna_new_data_function_for_score(new_data):
    global Moderna_country
    global Moderna_original
    global M_new2
    M_new1, M_new2 = pp.Preprocessing(new_data, Moderna_country)
    new_predict = Moderna_score_model.predict(M_new1)
    new_data['score'] = pd.Series(new_predict)
    Moderna_original = Moderna_original.append(new_data, ignore_index=True)

#실시간 크롤링시 이 함수를 호출해야
def Moderna_new_modeling(new_data):
    #Moderna_original에 행 100개 추가+score까지, M_new1, M_new2 생성
    global M_new2
    global Moderna_model
    global Moderna_embeddings
    global Moderna_original
    global Moderna_embeddings_backup
    global Moderna_original_backup
    try:
        Moderna_new_data_function_for_score(new_data)
        #NLP모델에 임베딩 추가

        new_embeddings = Moderna_model.encode(M_new2['text'])
        Moderna_embeddings = np.concatenate((Moderna_embeddings,new_embeddings),axis=0)
    except:
        Moderna_embeddings = Moderna_embeddings_backup.copy()
        Moderna_original = Moderna_original_backup.copy()


#user가 들어오면 실행하는 부분
def Moderna_qna(query):
    global Moderna_model
    global Moderna_embeddings
    global Moderna_original

    query_embedding = Moderna_model.encode(query)

    top_k=5
    cos_scores = util.pytorch_cos_sim(query_embedding, Moderna_embeddings)[0]
    cos_scores = cos_scores.cpu()
    #위의 진위성 점수랑 합하는 부분
    cos_scores = cos_scores + torch.tensor(Moderna_original['score'].values*0.2)
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(Moderna_original['user_name'].values[idx], " : ", Moderna_original['text'].values[idx], "(Score: %.4f)" % (score))
