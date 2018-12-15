from tika import parser
import os, sys
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import nltk
import gensim
import hdbscan


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

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(model)
print(cluster_labels)
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)

		
w1 = "amount"
print("AMOUNT: ")
print(model.wv.most_similar (positive=w1))

w2 = "peopl"
print("PEOPL: ")
print(model.wv.most_similar (positive=w2))

w3 = "human"
print("HUMAN: ")
print(model.wv.most_similar (positive=w3))

w4 = "comput"
print("COMPUT: ")
print(model.wv.most_similar (positive=w4))

w5 = "data"
print("DATA: ")
print(model.wv.most_similar (positive=w5))
				
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# fit a 2d PCA model to the vectors
#X = model[model.wv.vocab]
#pca = PCA(n_components=2)
#result = pca.fit_transform(X)
# create a scatter plot of the projection
#pyplot.scatter(result[:, 0], result[:, 1])
#words = list(model.wv.vocab)
#for i, word in enumerate(words):
#	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#pyplot.show()	

def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, len(model[word])), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()
	
display_closestwords_tsnescatterplot(model, 'comput')

display_closestwords_tsnescatterplot(model, 'peopl')