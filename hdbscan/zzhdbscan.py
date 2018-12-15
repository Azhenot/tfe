import hdbscan
import os, sys
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

files = os.listdir(os.getcwd())
documents = list()
				
for x in range (0, len(files)):
	if(os.path.isfile(files[x])):
		with open(files[x],"rt", encoding="utf8") as f:
			documents.append(f.read())
		print(files[x]+" DONE")
			

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names())
print(X.shape)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(X)
print(cluster_labels)
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)