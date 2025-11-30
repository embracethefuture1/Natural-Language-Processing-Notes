from sklearn.feature_extraction.text import CountVectorizer

#Documents

documents=[
    
    "Bu bir örnek metindir.",
    "Bu örnek metin doğal dil işlemeyi gösterir."
    
]

#unigram,bigram and trigram --> CountVectorizer

vectorizer_unigram=CountVectorizer(ngram_range=(1,1))
vectorizer_bigram=CountVectorizer(ngram_range=(2,2))
vectorizer_trigram=CountVectorizer(ngram_range=(3,3))

#unigram
X_Unigram=vectorizer_unigram.fit_transform(documents)
unigram_featurs=vectorizer_unigram.get_feature_names_out()


#bigram
X_Bigram=vectorizer_bigram.fit_transform(documents)
bigram_featurs=vectorizer_bigram.get_feature_names_out()



#trigram
X_trigram=vectorizer_trigram.fit_transform(documents)
trigram_featurs=vectorizer_trigram.get_feature_names_out()


print(trigram_featurs)
print(bigram_featurs)
print(unigram_featurs)
