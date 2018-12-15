

import pandas as pd
import pickle as pk
from scipy import sparse as sp

import os, sys
import logging


files = os.listdir(os.getcwd())
docs = list()
				
for x in range (0, len(files)):
	if(os.path.isfile(files[x])):
		with open(files[x],"rt", encoding="utf8") as f:
			docs.append(f.read())
		print(files[x]+" DONE")
		
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = docs[idx].replace("_", " ")
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]
    
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]
    
    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
  
    return docs
	
docs = docs_preprocessor(docs)

from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 10 times or more).
bigram = Phrases(docs, min_count=10)
trigram = Phrases(bigram[docs])

#for idx in range(len(docs)):
#    for token in bigram[docs[idx]]:
#        if '_' in token:
#            # Token is a bigram, add to document.
#            docs[idx].append(token)
#    for token in trigram[docs[idx]]:
#        if '_' in token:
#            # Token is a bigram, add to document.
#            docs[idx].append(token)
			
			
			
			
			
			
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)
print('Number of unique words in initital documents:', len(dictionary))

# Filter out words that occur less than 10 documents, or more than 20% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.2)
print('Number of unique words after removing rare and common words:', len(dictionary))

corpus = [dictionary.doc2bow(doc) for doc in docs]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 500 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 500
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)
					   
import pyLDAvis.gensim
visualisation = pyLDAvis.gensim.prepare(model, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_VisualizationAgain20_500.html')
