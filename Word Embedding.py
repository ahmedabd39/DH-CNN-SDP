from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pandas as pd 
import logging  # Setting up the loggings to monitor gensim
import warnings
import nltk

warnings.filterwarnings('ignore')
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
data = pd.read_csv("ant_1.7_with_nods.csv", encoding='utf-8')
text=data['nodes']
print(text[0])
# data.where(data['0'].notnull(), None)
sent = [row for row in text]
# print(sent)b
# all_words = [nltk.word_tokenize(row) for row in data['0']]
all_words = [nltk.word_tokenize(sntncs) for sntncs in sent]
model = Word2Vec(all_words, workers = 4, min_count = 1)
#Save word embedding model
model_file = 'ant_1.7_word2vec_embedding.txt'
model.wv.save_word2vec_format(model_file, binary=False)
print("model saved")
