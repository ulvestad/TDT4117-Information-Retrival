import random; random.seed(123)
import codecs
import re
import nltk
#nltk.download('all')
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
import gensim


#1.1 open and load the text file
f = codecs.open("pg3300.txt", "r", "utf-8")
doc = f.read()

#1.2 list of parahraps seperated by empty newline
blank_line_regex = r"(?:\r?\n){2,}"
paragraphs = re.split(blank_line_regex, doc.strip())

#1.3 filter out 'Gutenberg' paragraphs & 1.5 remove string punctiation + \r\t and lowercase 
filtered_paragrahps = []
for index, paragraph in enumerate(paragraphs):
    if "gutenberg" in paragraph.lower():
        continue
    else:
        filtered_paragrahps.append(paragraph.lower().translate(str.maketrans('','',string.punctuation+"\r\t")))

#1.4 tokenize the paragraphs =>  each paragraph is a list of words
#using NLTK Tokenizer Package which divides strings into lists of substrings.
processedParagraphs = [word_tokenize(word) for word in filtered_paragrahps]


#1.6 stem words
stemmer = PorterStemmer()
for i, paragraph in enumerate(processedParagraphs):
    for x, word in enumerate(paragraph):
        paragraph[x] = stemmer.stem(paragraph[x])

#print(processedParagraphs)
#print(len(processedParagraphs))

#2.1 Dictionary build
dictionary = gensim.corpora.Dictionary(processedParagraphs)

#2.2 Remove stopwords
stopwords_file = open('stopwords.txt')
words = stopwords_file.read()





