from bs4 import BeautifulSoup

# Takes plain text which is html
# Returns an array of tuples containing the (title, text)
def parseHtml(html):
	articles = []
	soup = BeautifulSoup(html, "lxml")

	arr = soup.find_all('doc')
	for doc in arr:
		articles.append((doc.attrs['title'], doc.text))
	
	return articles


lines = open('/Users/nickporter/Desktop/wiki_00', 'r').readlines()

# Call this method on all wik_xx files.
docs = parseHtml(''.join(lines))
