import os
import xml.etree.ElementTree as ET
import sys
import sqlite3
from scipy.sparse import csr_matrix
import numpy as np

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape = loader['shape'])


def get_trigram_idx(conn, gram):
    table_name = "trigrams"
    c = conn.cursor()
    c.execute("SELECT rowid FROM trigrams WHERE trigram = (?)", (gram,))
    r = c.fetchone()
    retval = None
    try:
        retval = r[0]
    except:
        print gram
    return retval


conn = sqlite3.connect('kgram_index.db')

BASEDIR = "k_grams"
OUTPUT_DIR = "trigram_vectors"

subdirs = os.listdir(BASEDIR)

indptr = [0]
indices = []
data = []
for subdir in subdirs:
    fns = sorted(os.listdir(os.path.join(BASEDIR, subdir)))
    key_dict = {}
    for fn in fns:
        full_path = os.path.join(BASEDIR, subdir, fn)
        with open(full_path) as fp:
            for gram in fp:
                gram = unicode(gram.strip(), "utf-8")
                index = get_trigram_idx(conn, gram)
                if not index:
                    continue
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

    id_path = os.path.join(OUTPUT_DIR, subdir)
    arr = csr_matrix((data, indices, indptr), dtype=int)
    save_sparse_csr(id_path, arr)

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
