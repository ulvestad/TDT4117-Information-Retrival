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


#2.1 Dictionary build and remove stopwords
dictionary = gensim.corpora.Dictionary(processedParagraphs)

stopwords_file = open('stopwords.txt')
words = stopwords_file.read()
stopwords = words.split(',')
stopwords_id_list = [dictionary.token2id[word] for word in stopwords if word in dictionary.token2id ]
dictionary.filter_tokens(stopwords_id_list)

#2.2 Create corpus (bags of words format)
corpus = [dictionary.doc2bow(paragraph) for paragraph in processedParagraphs]


#3.1 Build TF-IDF model using corpus
tfidf_model = gensim.models.TfidfModel(corpus)


#3.2 Map Bags-of-Words into TF-IDF weights
tfidf_corpus = tfidf_model[corpus]

#3.3 Construct MatrixSimilarity object
matrix_sim = gensim.similarities.MatrixSimilarity(corpus)

#3.4 Repeat for LSI model
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary,num_topics=100)
lsi_corpus = lsi_model[tfidf_corpus]
lsi_matrix = gensim.similarities.MatrixSimilarity(lsi_corpus)

#3.5 The first 3 LSI topics
tp1 = lsi_model.show_topic(1)
tp2 = lsi_model.show_topic(2)
tp3 = lsi_model.show_topic(3)
print('\nFirst 3 LSI topics:')
print(tp1)
print(tp2)
print(tp3)


#4.1 Query 
q = "What is the function of money?"
#q = "How taxes influence Economics?"
noPunctuationQuery = q.lower().translate(str.maketrans('','',string.punctuation+"\r\t"))
tokenizedQuery = [word_tokenize(noPunctuationQuery)][0] 
for x, word in enumerate(tokenizedQuery):
    tokenizedQuery[x] = stemmer.stem(tokenizedQuery[x])
query = dictionary.doc2bow(tokenizedQuery)


#4.2 Convert BOW to TF-IDF representation. 
query_tfidf = tfidf_model[query]
print("\nTF_IDF representation: ")
for weight in query_tfidf:
    print(dictionary.get(weight[0]), weight[1])


#4.3 Report top 3 the most relevant paragraphs for the query
index = gensim.similarities.MatrixSimilarity(tfidf_corpus)
docs2similarity = enumerate(index[query_tfidf])
retrievedDocs = sorted(docs2similarity, key=lambda kv: -kv[1])[:3]
print("\nTop 3 TF-IDF's: ")
for doc in retrievedDocs:
    print("Paragraph: ",doc[0]+1)


#4.4 Convert query TF-IDF representation into LSI-topics representation
lsi_query = lsi_model[query]
sorted_li = (sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3] )
all_topics = lsi_model.show_topics()
print("\nMost relevant paragraphs: ")
for para in sorted_li:
    print("topic:", para[0])
    print((all_topics[para[0]][1]))
