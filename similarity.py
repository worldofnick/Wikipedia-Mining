import os
import sys
import numpy as np
import csv
from scipy.sparse import csr_matrix
from collections import defaultdict

'''
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape = loader['shape'])

arr = load_sparse_csr(os.path.join('trigram_vectors','AB.npz'))
print arr
print arr.shape

'''

def jaccardSimilarity(s1, s2):
    js = len(s1 & s2)/float(len(s1 | s2))
    return js

BASEDIR = "k_grams_no_punc"
OUT_FILE = "article_similarities.txt"
DICT_DIR = "raw_text"

subdirs = os.listdir(BASEDIR)

for subdir in subdirs[0:1]:
    simdict = defaultdict(list)
    simlist = []
    sim_values = set()
    set_compared = set()
    dict_path = os.path.join(DICT_DIR, subdir, 'doc_ids.key')
    reader = csv.reader(open(dict_path), delimiter = '\t')
    doc_ids = dict(reader)
    fns = sorted(os.listdir(os.path.join(BASEDIR, subdir)))
    for i, fn1 in enumerate(fns):
        full_path1 = os.path.join(BASEDIR, subdir, fn1)
        s1 = set([line.strip('\n') for line in open(full_path1)])
        for j, fn2 in enumerate(fns):
            if fn1 == fn2:
                continue
            if (fn1, fn2) in set_compared or (fn2, fn1) in set_compared:
                continue
            full_path2 = os.path.join(BASEDIR, subdir, fn2)
            s2 = set([line.strip('\n') for line in open(full_path2)])
            sim = jaccardSimilarity(s1, s2)
            art1 = doc_ids[fn1.strip('_kgrams.txt')]
            art2 = doc_ids[fn2.strip('_kgrams.txt')]
            #simlist.append([art1, art2, sim])
            sim_values.add(sim)
            simdict[str(sim)].append([art1, art2])
            set_compared.add((fn1, fn2))
    
    for key in sorted(list(sim_values), reverse=True):
        for articles in simdict[str(key)]:
            simlist.append([key] + articles)


    with open( OUT_FILE, "w") as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(simlist)
    fp.close()

