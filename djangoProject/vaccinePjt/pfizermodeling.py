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

Pfizer_country = ''
Pfizer_original = ''

data1_pfizer,data2_pfizer = '',''
Pfizer_score_model = ''
Pfizer_model = ''
Pfizer_embeddings = ''

Pfizer_embeddings_backup = ''
Pfizer_original_backup = ''


P_new2 = ''

def forPfizerInit():
    global Pfizer_country
    global Pfizer_original
    Pfizer_country = pd.read_csv('Pfizer_country.csv')
    Pfizer_original = pd.read_csv('vaccination_tweets.csv')
    global data1_pfizer
    global data2_pfizer
    data1_pfizer,data2_pfizer = pp.preprocess_all(Pfizer_original, Pfizer_country)

    global Pfizer_score_model
    Pfizer_score_model = KMeans(n_clusters=2).fit(data1_pfizer)
    Pfizer_original['score'] = Pfizer_score_model.labels_

    global Pfizer_model
    global Pfizer_embeddings
    global Pfizer_embeddings_backup
    global Pfizer_original_backup
    Pfizer_model=SentenceTransformer('bert-large-nli-mean-tokens')
    Pfizer_embeddings = Pfizer_model.encode(data2_pfizer['text'])
    Pfizer_embeddings_backup = Pfizer_embeddings.copy()
    Pfizer_original_backup = Pfizer_original.copy()
    print("Server successfull loaded")

def Pfizer_new_data_function_for_score(new_data):
    global Pfizer_country
    global Pfizer_original
    global P_new2
    P_new1, P_new2 = pp.Preprocessing(new_data, Pfizer_country)
    new_predict = Pfizer_score_model.predict(P_new1)
    new_data['score'] = pd.Series(new_predict)
    Pfizer_original = Pfizer_original.append(new_data, ignore_index=True)

#실시간 크롤링시 이 함수를 호출해야
def Pfizer_new_modeling(new_data):
    #Pfizer_original에 행 100개 추가+score까지, P_new1, P_new2 생성
    global P_new2
    global Pfizer_model
    global Pfizer_embeddings
    global Pfizer_original
    global Pfizer_embeddings_backup
    global Pfizer_original_backup
    try:
        Pfizer_new_data_function_for_score(new_data)
        #NLP모델에 임베딩 추가

        new_embeddings = Pfizer_model.encode(P_new2['text'])
        Pfizer_embeddings = np.concatenate((Pfizer_embeddings,new_embeddings),axis=0)
    except:
        Pfizer_embeddings = Pfizer_embeddings_backup.copy()
        Pfizer_original = Pfizer_original_backup.copy()


#user가 들어오면 실행하는 부분
def Pfizer_qna(query):
    global Pfizer_model
    global Pfizer_embeddings
    global Pfizer_original

    query_embedding = Pfizer_model.encode(query)

    top_k=5
    cos_scores = util.pytorch_cos_sim(query_embedding, Pfizer_embeddings)[0]
    cos_scores = cos_scores.cpu()
    #위의 진위성 점수랑 합하는 부분
    cos_scores = cos_scores + torch.tensor(Pfizer_original['score'].values*0.2)
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(Pfizer_original['user_name'].values[idx], " : ", Pfizer_original['text'].values[idx], "(Score: %.4f)" % (score))
