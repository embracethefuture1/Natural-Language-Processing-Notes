import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk 
from nltk.corpus import stopwords

df=pd.read_csv("C:\\Users\\oguzz\\Desktop\\BagOfWords_IMBD_DataSet\\IMDB_Dataset.csv")

df2=df.head(100)

 
# Text Data Retrieval

documents=df2["review"]
labels=df2["sentiment"] # Possitive or Negative

# Text Cleaning Function
 
def clean_text(text):
    
     
     # LowerWord Transform
     
     text=text.lower()
     
     # Number Cleaning
     
     text=re.sub(r"\d+","",text)
     
     # Special Char Cleaning
     
     text=re.sub(r"[^\w\s]","",text)
     
     # Shorts Words Cleaning
     
     text=" ".join([word for word in text.split() if len(word)>2])
     
     # Stop Words Cleaning

     stop_words_eng=set(stopwords.words("english"))
     Filtered_Words=[word for word in text.split() if word.lower() not in stop_words_eng]
     
     return " ".join(Filtered_Words)
 
 
cleaned_documents=[clean_text(doc) for doc in documents]
 
#print(cleaned_documents)

# bow
vectorizer = CountVectorizer()

# Text -> Number Vector

X=vectorizer.fit_transform(cleaned_documents[:100])

# Word Clustering

feature_names=vectorizer.get_feature_names_out()

# Vector Representation

#print("Vektör Temsili:")
vector_representation2=X.toarray()[:2]

# Vector Representation Dataframe

df_bow=pd.DataFrame(X.toarray(),columns=feature_names)

# Word Frequency

word_counts=X.sum(axis=0).A1
word_freq=dict(zip(feature_names,word_counts))
#print(word_freq)

# First 5 Words

most_common_words=Counter(word_freq).most_common(5)

#print("En çok tekrar eden 5 kelime :",most_common_words)

print(cleaned_documents)