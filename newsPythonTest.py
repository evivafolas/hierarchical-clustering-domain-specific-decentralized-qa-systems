import json
import newspaper
from newspaper import Article
from requests import request
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import fulltext
import spacy
import nltk
from collections import Counter
from string import punctuation

nlp = spacy.load("el_core_news_sm")

def get_hotwords(text):
    
    result =[]

    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())

    for token in doc:
        if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)

    return result


"""
url = 'http://fox13now.com/2013/12/30/new-year-new-laws-obamacare-pot-guns-and-drones/'

article = Article(url)

article.download()
article.parse()

print(article.summary)

"""

cnn_paper = newspaper.build('http://cnn.com')

# for article in cnn_paper.articles:
#     print(article.url)

# for category in cnn_paper.category_urls():
#     print(category)

# html = requests.get('https://edition.cnn.com/2022/11/16/asia/g20-summit-day-2-russia-intl-hnk/index.html').text
# text = fulltext(html)

url = 'https://www.kathimerini.gr/world/562140322/tramp-exi-logoi-gia-toys-opoioys-to-kynigi-tis-proedrias-tha-einai-pio-dyskolo-tora/'

a1 = Article(url, language='el')

a1.download()
a1.parse()
a1.nlp()

print(a1.keywords)
print(a1.summary)

# print(a1.authors)
# print(a1.title + '\n' + a1.text[:])

# testJson = json.dumps(a1)

text = a1.text

# doc = nlp(text)

# print(doc.ents)

hotwords = set(get_hotwords(text))
most_common_list = Counter(hotwords).most_common(10)

for item in most_common_list:
    print(item[0])
