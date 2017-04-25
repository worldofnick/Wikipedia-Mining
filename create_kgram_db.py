import sqlite3
conn = sqlite3.connect('kgram_index.db')

table_name = "trigrams"

c = conn.cursor()

# Create table
sql = 'create table if not exists ' + table_name + ' (id integer, trigram text)'
c.execute(sql)

sql = "CREATE INDEX index_name ON trigrams (trigram)"
c.execute(sql)

with open("uniq_k_gram_index.txt") as fp:

    for i, line in enumerate(fp):
        # Insert a row of data
        tgram = unicode(line.strip(), "utf-8")
        c.execute("INSERT INTO trigrams(id, trigram) VALUES (?,?)", (i, tgram))

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
