import os
import sys

BASEDIR = "k_grams"

subdirs = os.listdir(BASEDIR)

with open("k_gram_index.txt", "w") as out_fp:
    for subdir in subdirs:
        fns = os.listdir(os.path.join(BASEDIR, subdir))
        for fn in fns:
            full_path = os.path.join(BASEDIR, subdir, fn)
            with open(full_path) as in_fp:
                grams = set((unicode(item.strip(), "utf-8") for item in in_fp.readlines()))
            for gram in grams:
                out_fp.write("%s\n" % gram.encode("utf-8"))
