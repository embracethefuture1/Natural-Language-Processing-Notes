#Kütüphanelerin içeriye aktarılması

import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re


#veri setinin yüklenmesi

df=pd.read_csv("C:\\Users\\oguzz\\Documents\\GitHub\\Natural-Language-Processing-Notes\\Word_Embedding_IMDB_DataSet\\IMDB_Dataset.csv")
df2=df.head(100)
print(df.head(5))

document=df2["review"]

#metin temizlenmesi

def clean_text(text):
    text=text.lower()# Kucuk Harf
    text=re.sub(r"\d+","",text)# SayılarI Temizle
    text=re.sub(r"^\w\s","",text)# ozel karakterleri temizle
    text=" ".join(word for word in text.split() if len(word)>2)# Kısa kelimeleri temizle
    return text

cleaned_documents=[clean_text(doc) for doc in document]
    

#cümleleri tokenization islemi

tokenized_documents=[simple_preprocess(doc) for doc in cleaned_documents]

print(tokenized_documents)

#word2vec modeli tanımlayalım

model=Word2Vec(sentences=tokenized_documents,vector_size=50,window=1,sg=0)
word_vectors=model.wv
words=list(word_vectors.index_to_key)[:400]
vectors=[word_vectors[word] for word in words]

#clustering 2 veya 3 adet kume olusturma

kmeans=KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters=kmeans.labels_


#pca 50->2

pca=PCA(n_components=2)
reduce_vectors=pca.fit_transform(vectors)

#2d gorsellestirme

plt.figure()
plt.scatter(reduce_vectors[:,0],reduce_vectors[:,1],c=clusters,cmap='viridis')

centers=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],cmap="red",marker="x",s=120,label="Merkez")
plt.legend()


#figure uzerine kelime ekleme islemi
for i, word in enumerate(words):
    plt.text(reduce_vectors[i,0],reduce_vectors[i,1],word,fontsize=8)
    
plt.title("Word2Vec")    
plt.show()