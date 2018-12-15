import os, sys
import logging


files = os.listdir(os.getcwd())
documents = list()
				
for x in range (0, len(files)):
	if(os.path.isfile(files[x])):
		with open(files[x],"rt", encoding="utf8") as f:
			documents.append(f.read())
		print(files[x]+" DONE")

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

# loop through document list
for i in documents:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
print("test")
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=20, num_words=3))

import pyLDAvis.gensim
visualisation = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

