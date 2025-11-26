from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


# Example Documentry
documents=["Kedi çok tatlı bir hayvandır",
           "Kedi ve köpekler çok tatlı hayvanlardır",
           "Arılar bal üretirler"
           ]

tfidf_vectorizer=TfidfVectorizer()

# Text --> Number

X=tfidf_vectorizer.fit_transform(documents)

# Word Cluster

feature_names=tfidf_vectorizer.get_feature_names_out()
print("TF-IDF Vektör Temsilleri :",)
Vector_Representation=X.toarray()
print(Vector_Representation)

print(feature_names)


df_tfidf=pd.DataFrame(Vector_Representation,columns=feature_names)

kedi_tfidf=df_tfidf["kedi"]
kedi_mean_tfidf=np.mean(kedi_tfidf)
print(kedi_mean_tfidf)

arilar_tfidf=df_tfidf["arılar"]
arilar_mean_tfidf=np.mean(arilar_tfidf)
print(arilar_mean_tfidf)
