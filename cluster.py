# Author: Nick Porter - University of Utah School of Computing

# Takes raw text files of wikipedia artciles and vectorizes them using tf-idf.
# Reduces the vectors into 2 dimensional and plots them
# Clusters the high dimensional data and retrives the most popular terms per cluster using k-Means
# Clusters and plots the 2D data using k-Means
# Using mean shift clustering to estimate the number of clusters in the data set and plots it

import matplotlib
matplotlib.use('Agg')
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle

import logging
from optparse import OptionParser
import sys
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

def make_corpus(root):

  for path, subdirs, files in os.walk(root):
    count = 0
    for name in files:
      file = open(path + '/' + name, 'r')
      count += 1
      if count > 500:
        break
      text = file.read()
      yield text
    
def vectorize_articles(root):
  vectorizer = TfidfVectorizer(max_df=0.5, max_features=None,
                                 min_df=2, stop_words='english',
                                 use_idf=True, lowercase=True)

  corpus = make_corpus(root)
  X = vectorizer.fit_transform(corpus)
  return vectorizer, X

# Use this function on high dimensional data
def cluster(X, k):
  km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, verbose=False)
  km.fit(X)
  return km

# X is high dimensional data
def get_cluster_stats(X, km, vectorizer, k):

  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, km.labels_, sample_size=1000))

  print("Top terms per cluster:")
  order_centroids = km.cluster_centers_.argsort()[:, ::-1]

  terms = vectorizer.get_feature_names()
  for i in range(k):
      print("Cluster %d:" % i)
      string = ''
      for ind in order_centroids[i, :10]:
        string = '%s %s' % (string, terms[ind])

      print(string)
      print()

def reduce_data(n_components, X):
  # try svd
  reduced_data = TruncatedSVD(n_components=n_components).fit_transform(X)
  return reduced_data

# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1, verbose=False)
# kmeans.fit(reduced_data)

def plot_reduced_data(reduced_data):
  
  for i in range(0, reduced_data.shape[0]):
  	plt.scatter(reduced_data[i,0], reduced_data[i,1], c='r', marker='*')
  plt.savefig('article_plot.pdf')
  print 'Saved article_plot.pdf'

def plot_reduced_clusters(reduced_data, kmeans):

  h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

  # Plot the decision boundary. For that, we will assign a color to each
  x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
  y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Obtain labels for each point in mesh. Use last trained model.
  Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.figure(1)
  plt.clf()
  plt.imshow(Z, interpolation='nearest',
             extent=(xx.min(), xx.max(), yy.min(), yy.max()),
             cmap=plt.cm.Paired,
             aspect='auto', origin='lower')

  plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
  # Plot the centroids as a white X
  centroids = kmeans.cluster_centers_
  plt.scatter(centroids[:, 0], centroids[:, 1],
              marker='x', s=169, linewidths=3,
              color='w', zorder=10)
  plt.title('K-means clustering on the Wikipedia Dataset dataset (PCA-reduced data)\n'
            'Centroids are marked with white cross')
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  plt.xticks(())
  plt.yticks(())
  plt.show()

  plt.savefig('clusters2.pdf')
  print 'Saved cluster2.pdf'

def mean_shift(X):
  bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
  ms.fit(X)
  labels = ms.labels_
  cluster_centers = ms.cluster_centers_

  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)

  colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
  for k, col in zip(range(n_clusters_), colors):
      my_members = labels == k
      cluster_center = cluster_centers[k]
      plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
      plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
               markeredgecolor='k', markersize=14)
  plt.title('Estimated number of clusters: %d' % n_clusters_)
  plt.savefig('meanshift.pdf')
  print 'Saved meanshift.pdf'
  print("number of estimated clusters : %d" % n_clusters_)

def main():
  root = '/home/shared/Wikipedia-Mining/raw_text/'

  # Data pre-processing
  print 'Vectorizing articles'
  vectorizer, X = vectorize_articles(root)
  # X = pickle.load( open( "vector_articles.pickle", "rb" ) )
  # vectorizer = pickle.load( open( "vectorizer.pickle", "rb" ) )

  # Save vectorized articles for easy re-use
  with open('vector_articles.pickle', 'wb') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open('vectorizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Cluster in high dimensions
  print 'Clustering in high dimensions'
  km = cluster(X, 12)

  # Get cluster stats
  print 'Getting cluster stats'
  get_cluster_stats(X, km, vectorizer, 12)

  # Scale down to 2 dimensions
  print 'Reducing Data'
  reduced_data = reduce_data(2, X)

  # Visualize reduced data
  print 'Plotting reduced Data'
  plot_reduced_data(reduced_data)

  # Cluster and visualize 2d clusters
  print 'Clustering in 2D'
  km = cluster(reduced_data, 4)

  print 'Plotting reduced clusters'
  plot_reduced_clusters(reduced_data, km)

  # Estimate the number of clusters using mean shift clustering
  print 'Mean Shift Clustering'
  mean_shift(reduced_data)

main()
