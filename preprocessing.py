import pandas as pd
from konlpy.tag import Okt
import re
from tqdm import tqdm
df = pd.read_csv('./crawling_data/lol_champ_repl_new.csv',names=['name','review'],encoding='cp949')
print(df.head(10))
df.info()

okt = Okt()

df_stopwords = pd.read_csv('./stopwords.csv')
stopwords = list(df_stopwords['stopword'])
cleaned_sentences = []
for review in tqdm(df.review):
    review = re.sub('[^가-힣]',' ',review)
    tokened_review = okt.pos(review, stem=True)
    #print(tokened_review)

    df_token = pd.DataFrame(tokened_review, columns=['word','class'])
    df_token = df_token[(df_token['class']=='Noun')|
                        (df_token['class']=='Verb')|
                        (df_token['class']=='Adjective')
    ]
    #print(df_token.head())
    words = []
    for word in df_token.word:
        if len(word)>0:
            if not word in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)

df['reviews'] = cleaned_sentences
df = df[['name','reviews']]
print(df.head(10))
df.info()
df.to_csv('./crawling_data/cleaned_lol_review.csv', index=False)