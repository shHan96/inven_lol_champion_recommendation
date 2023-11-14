import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
import re
from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore,key = lambda x: x[1],reverse=True)
    simScore = simScore[:11]
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movieIdx, 0]
    return recMovieList

df_reviews = pd.read_csv('./crawling_data/cleaned_review.csv')
Tfidf_matrix = mmread('./models/Tfidf_lol_review.mtx').tocsr()
with open('./models/tfidf_one.pickle','rb') as f:
    Tfidf = pickle.load(f)
# print(df_reviews.iloc[104,0])
# cosine_sim = linear_kernel(Tfidf_matrix[104],Tfidf_matrix)
# print(cosine_sim[0])
# print(len(cosine_sim[0]))
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)
# try:
#     embedding_model = Word2Vec.load('./models/word2vec_lol_review.model')
#     keyword = '씨에스'
#     sim_word = embedding_model.wv.most_similar(keyword, topn=10)
#     print(sim_word)
#     words = [keyword]
#     for word, _ in sim_word:
#         words.append(word)
#     print(words)
#
#     # 가장 많이 나온 단어는 10개 그 다음은 9개 그 다음은 8개 ...
#     sentence = []
#     count = 10
#     for word in words:
#         sentence = sentence + [word] * count
#         count-=1
#     sentence = ' '.join(sentence)
#     print(sentence)
#     sentence_vec = Tfidf.transform([sentence])
#     cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
#     recommendation = getRecommendation(cosine_sim)
#     print(recommendation)
# except:
#     print('다른 키워드를 이용하세요.')


sentence = '과학'
okt = Okt()
df_stopwords = pd.read_csv('./stopwords.csv')
stopwords = list(df_stopwords['stopword'])
sentence = re.sub('[^가-힣]',' ',sentence)
tokened_sentence = okt.pos(sentence, stem=True)
print(tokened_sentence)
sentence = [i[0] for i in tokened_sentence if i[1]=='Noun' or i[1]=='Adjective' or i[1]=='Verb']
words = []
for word in sentence:
    if len(word) > 0:
        if not word in stopwords:
            words.append(word)
cleaned_sentence = ' '.join(words)
print(cleaned_sentence)
try:
    sentence_vec = Tfidf.transform([cleaned_sentence])
    cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
    recommendation = getRecommendation(cosine_sim)
    print(recommendation)
except:
    print('다른 키워드를 이용하세요.')