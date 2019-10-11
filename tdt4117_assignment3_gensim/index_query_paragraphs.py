import random; random.seed(123)
import codecs
import re
import nltk
#nltk.download('all')
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


#1.1 open and load the text file
f = codecs.open("pg3300.txt", "r", "utf-8")
doc = f.read()

#1.2 list of parahraps seperated by empty newline
blank_line_regex = r"(?:\r?\n){2,}"
paragraphs = re.split(blank_line_regex, doc.strip())

#1.3 filter out 'Gutenberg' paragraphs and 1.5 remove string punctiation + \r\n and lowercase 
filtered_paragrahps = []
for index, paragraph in enumerate(paragraphs):
    if "gutenberg" in paragraph.lower():
        continue
    else:
        filtered_paragrahps.append(paragraph.lower().translate(str.maketrans('','',string.punctuation+"\n\r\t")))

#1.4 tokenize the paragraphs =>  each paragraph is a list of words
#using NLTK Tokenizer Package which divides strings into lists of substrings.
tokenized = [word_tokenize(word) for word in filtered_paragrahps]


stemmer = PorterStemmer()
word = "Economics"
print(stemmer.stem(word.lower()))

#print(tokenized)
#print(len(filtered_paragrahps))