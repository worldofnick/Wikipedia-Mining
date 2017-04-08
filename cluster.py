import matplotlib
matplotlib.use('Agg') 

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np

articles = []
# Load articles into array
# root = '/home/crachmanin/Wikipedia-Mining/raw_text/'

# for path, subdirs, files in os.walk(root):
# 	for name in files:
# 		file = open(path + '/' + name, 'r')
# 		articles.append(file.read())

path = '/home/crachmanin/Wikipedia-Mining/raw_text/AA/'
files = os.listdir(path)
# print 'Reading files'
for file_name in files:
	file = open(path + file_name, 'r')
	articles.append(file.read())

vectorizer = TfidfVectorizer(max_df=0.5, max_features=None,
                                 min_df=2, stop_words='english',
                                 use_idf=True, lowercase=True)

print 'Vectorizing files'
X = vectorizer.fit_transform(articles)

# print 'Fitting with k-means'
k = 4
km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)

# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()


# k_scores = []
# for i in range(2, 50):
# 	km = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1, verbose=False)
# 	km.fit(X)
# 	k_scores.append(metrics.silhouette_score(X, km.labels_, sample_size=1000))
# 	print 'Added score for %d' % i

# print k_scores
# plt.plot(k_scores)
# plt.savefig('kscores.pdf')

# try svd
# reduced_data = TruncatedSVD(n_components=2).fit_transform(X)

# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1, verbose=False)
# kmeans.fit(reduced_data)

# # shapes = ['*', 'v', '^', '.']
# # print reduced_data.shape
# # for i in range(0, reduced_data.shape[0]):
# # 	plt.scatter(reduced_data[i,0], reduced_data[i,1], c='b', marker='*')

# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# plt.savefig('clusters1.png')


