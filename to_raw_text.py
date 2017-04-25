import os
import xml.etree.ElementTree as ET
import sys

def write_file(path, text):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(path, "w") as f:
        f.write(text)

BASEDIR = "extracted"
OUTPUT_DIR = "raw_text"

subdirs = os.listdir(BASEDIR)

for subdir in subdirs:
    fns = os.listdir(os.path.join(BASEDIR, subdir))
    key_dict = {}
    for fn in fns:
        full_path = os.path.join(BASEDIR, subdir, fn)
        with open(full_path) as fp:
            data = fp.read()
            articles = [item.strip() for item in data.split("</doc>")][:-1]
            for article in articles:
                header = article[:article.find('\n')].strip() + "</doc>"
                text = unicode(article[article.find('\n'):].strip(), "utf-8")
                try:
                    root = ET.fromstring(header)
                    doc_id = root.attrib["id"]
                    title = root.attrib["title"]
                    key_dict[doc_id] = title
                    new_fn = doc_id + ".txt"
                    new_path = os.path.join(OUTPUT_DIR, subdir, new_fn)
                    write_file(new_path, text.encode("utf-8"))
                except Exception as e:
                    print full_path, path
                    sys.exit()

    id_string = '\n'.join(["%s\t%s" % pair for pair in key_dict.items()])
    id_fn = "doc_ids.key"
    id_path = os.path.join(OUTPUT_DIR, subdir, id_fn)
    write_file(id_path, id_string.encode("utf-8"))
