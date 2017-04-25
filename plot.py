import matplotlib as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape = loader['shape'])

sparse_matrix = load_sparse_csr('/home/crachmanin/Wikipedia-Mining/trigram_vectors/AS.npz')
small_matrix = sparse_matrix[:10,:]
reduced_data = TruncatedSVD(n_components=2).fit_transform(small_matrix)

plt.title('Article vectors recuded to d=2 by PCA')
plt.plot(X[:, 0], X[:, 1])
plt.savefig('pca.pdf')