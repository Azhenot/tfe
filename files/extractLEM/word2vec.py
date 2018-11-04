from tika import parser
import os, sys
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import nltk
import gensim

pos_to_wornet_dict = {

    'JJ': wn.ADJ,
    'JJR': wn.ADJ,
    'JJS': wn.ADJ,
    'RB': 'a',
    'RBR': 'a',
    'RBS': 'a',
    'NN': 'n',
    'NNP': 'n',
    'NNS': 'n',
    'NNPS': 'n',
    'VB': 'v',
    'VBG': 'v',
    'VBD': 'v',
    'VBN': 'v',
    'VBP': 'v',
    'VBZ': 'v',
    'DT' : 'n',
    'IN' : 'n',
    'PRP$' : 'n',
    'MD' : 'n',
    'TO' : 'n',
    'CC' : 'n',
    'PRP' : 'n'

}


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

files = os.listdir(os.getcwd())

class MySentences(object):
	def __iter__(self):
		for x in range (0, len(files)):
			if os.path.isfile(files[x]):
				with open(files[x],encoding="utf8") as f:
					for line in f:
						yield line.split()
			print(files[x]+" DONE")
				
sentences = MySentences()

model = gensim.models.Word2Vec(
	sentences,
	size=150,
	window=10,
	min_count=2,
	workers=10)
model.train(sentences, total_examples=1, epochs=10)
print(len(model.wv.vocab))

		
w1 = "amount"
print("AMOUNT: ")
print(model.wv.most_similar (positive=w1))

w2 = "people"
print("PEOPL: ")
print(model.wv.most_similar (positive=w2))

w3 = "human"
print("HUMAN: ")
print(model.wv.most_similar (positive=w3))

w4 = "computer"
print("COMPUT: ")
print(model.wv.most_similar (positive=w4))

w5 = "data"
print("DATA: ")
print(model.wv.most_similar (positive=w5))
				
				
				
				
				