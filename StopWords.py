import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# Stop Words List Upload For English

stop_words_eng=set(stopwords.words("english"))

# Example Text Engilish

text="This is an example of removing stop words from a text document."
Filtered_Words=[word for word in text.split() if word.lower() not in stop_words_eng]
print("Filtered Words :",Filtered_Words)



# Stop Words List Upload For Turkish

stop_words_eng=set(stopwords.words("turkish"))

# Example Text Turkish

text="Bu örnek sana ve diğer insanlara durdurma kelimelerinin mantığını bir bağlamda gösterecektir ve bu bağlamda şu an gördüğün örnekte yer alıyor."
Filtered_Words=[word for word in text.split() if word.lower() not in stop_words_eng]
print("Filtered Words :",Filtered_Words)

#We Make Stop Words

Turkish_StopWords=set(["ve","bir","bu","ile"])

#Example Text

text="Bu bir örnek metin ve stop words'leri temizlemek için kullanılıyor."
Filtered_Words=[word for word in text.split() if word.lower() not in Turkish_StopWords]
print("Filtered Words :",Filtered_Words)