#Word Tokenization 

import nltk

nltk.download("punkt_tab")

text="Hello, World Oguzhan! How are you"

word_tokens=nltk.word_tokenize(text)

print(word_tokens)


#Sentence Tokenization

sentenc_tokens=nltk.sent_tokenize(text)

print(sentenc_tokens)