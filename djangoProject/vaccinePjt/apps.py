from django.apps import AppConfig
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch


Pfizer_original = ''
Moderna_original = ''
Pfizer_embeddings = ''
Moderna_embeddings = ''
model = ''

def forInit():
    global Pfizer_original
    global Moderna_original
    global Pfizer_embeddings
    global Moderna_embeddings
    global model

    model=SentenceTransformer('msmarco-distilbert-base-v2')

    Pfizer_original = pd.read_csv('Pfizer_original_withscore.csv')
    Moderna_original = pd.read_csv('Moderna_original_withscore.csv')
    #저장할때 얘를 헤더 없이 저장해서 read할때 header=None 꼭 있어야됨!
    Pfizer_embeddings = pd.read_csv('Pfizer_embeddings.csv',header = None)
    Moderna_embeddings = pd.read_csv('Moderna_embeddings.csv', header = None)

#user가 들어오면 실행하는 부분
def qna(original,embeddings):
  query=input('Enter the query here :')
  query_embedding = model.encode(query)

  top_k=5
  cos_scores = util.pytorch_cos_sim(query_embedding, torch.tensor(embeddings.values.astype(np.float32)))[0]
  cos_scores = cos_scores.cpu()
  #위의 진위성 점수랑 합하는 부분
  cos_scores = cos_scores + torch.tensor(original['score'].values*0.2)
  top_results = torch.topk(cos_scores, k=top_k)

  print("\n\n======================\n\n")
  print("Query:", query)
  print("\nTop 5 most similar sentences in corpus:")

  for score, idx in zip(top_results[0], top_results[1]):
      print(original['user_name'].values[idx], " : ", original['text'].values[idx], "(Score: %.4f)" % (score))


class VaccinepjtConfig(AppConfig):
    name = 'vaccinePjt'

    def ready(self):
        forInit()
        qna(Pfizer_original,Pfizer_embeddings)

        pass
