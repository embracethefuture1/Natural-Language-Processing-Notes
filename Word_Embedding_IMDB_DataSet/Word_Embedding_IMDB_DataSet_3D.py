#Kütüphanelerin içeriye aktarılması

import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D


#veri setinin yüklenmesi

df=pd.read_csv("C:\\Users\\oguzz\\Documents\\GitHub\\Natural-Language-Processing-Notes\\Word_Embedding_IMDB_DataSet\\IMDB_Dataset.csv")
df2=df.head(100)
print(df.head(5))

document=df2["review"]

#metin temizlenmesi

def clean_text(text):
    text=text.lower()# Kucuk Harf
    text=re.sub(r"\d+","",text)# SayılarI Temizle
    text=re.sub(r"[^\w\s]","",text)# ozel karakterleri temizle
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

kmeans=KMeans(n_clusters=3)
kmeans.fit(vectors)
clusters=kmeans.labels_


#pca 50->3

pca=PCA(n_components=3)
reduce_vectors=pca.fit_transform(vectors)

#3d gorsellestirme


fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection="3d")

ax.scatter(reduce_vectors[:,0],reduce_vectors[:,1],reduce_vectors[:,2],cmap='viridis',c=clusters,s=20)

centers=pca.transform(kmeans.cluster_centers_)
ax.scatter(centers[:,0],centers[:,1],centers[:,2],color="red",marker="x",s=120,label="Merkez")
ax.legend()


#figure uzerine kelime ekleme islemi
for i, word in enumerate(words):
    ax.text(reduce_vectors[i,0],reduce_vectors[i,1],reduce_vectors[i,2],word,fontsize=8)
    
plt.title("Word2Vec")    
plt.show()