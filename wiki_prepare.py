# Takes all the raw .txt files from wikipedia and converts each article into a set of kgrams

from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidTransformer
import glob

def word_kgram(text, k):
	grams = ngrams(text, k)
	gram_set = set()
	for gram in grams:
		gram_set.add(' '.join(gram))
	return gram_set

def vectorizeArticle(text):
	# Remove stop words
	words = nltk.word_tokenize(text)
	stops = set(stopwords.words("english"))
	filtered_article = [word for word in words if word not in stops]

	# k-grams
	count = CountVectorizer()
	bagOfWords = count.fit_transform(text, ngram_range=(3,3))

	# term frequency
	tfidf = TfidTransformer()
	return tfidf.fit_transform(bagOfWords).toarray()


def saveSet(kgrams, articleName):
	# save as a text file
	f = open('%s_kgrams.txt' % articleName, "w")
	f.write(",".join(map(lambda x: str(x), kgrams)))
	f.close()

# Iterate each file in your dir, vectorize using kgrams and save the set
textFiles = glob.glob('/somePath/*.txt') # List of all text files
for filename in textFiles:
	file = open(filename)
	articleKgrams = vectorizeArticle(file.read())
	saveSet(articleKgrams, filename)
