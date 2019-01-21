# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer


# from nltk.tokenize import word_tokenize 
# import nltk
def getTrainText(raw):
  data = []
  labels = []
  for x in raw.split("\n")[1:]:
    if (len(x) > 0 ):
      data.append(x.split("\t")[0])
      labels.append(x.split("\t")[1])
  return data , labels

def cleanData(data):
  for i in range (len(data)):
    # data[i] = re.sub("[^a-zA-Z]"," ", data[i])
    data[i] = data[i].lower()
    data[i]  = re.sub(r"\\", "", data[i] )    
    data[i]  = re.sub(r"\'", "", data[i] )    
    data[i]  = re.sub(r"\"", "", data[i] )    
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    data[i] = data[i].translate(translate_map)

  return data  

# print (os.listdir("."))
vectorizer = CountVectorizer(ngram_range = (1,1),max_df = 0.1)

with open('./training.txt', 'r') as content_file:
    raw = content_file.read()
# print (raw.split("\n"))
data,labels = getTrainText(raw)
# print (data,labels)  
n = int (input())
data2 = []
for i in range(n):
    data2.append(input())

data = cleanData(data)
data2 = cleanData(data2)
# print (data)
gnb = GaussianNB()
train = vectorizer.fit_transform(data)  

test = vectorizer.transform(data2)

tfidf_vectorizer = TfidfTransformer()
tfidf_matrix = tfidf_vectorizer.fit_transform(train)

tfidf_matrix2 = tfidf_vectorizer.fit_transform(test)
y_pred = gnb.fit(tfidf_matrix.toarray(), labels).predict(tfidf_matrix2.toarray())

for x in y_pred:
    print (x)
# print (y_pred)
# print (data2)
# for x in data2:
#     data.append(x)
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(data)
#     data.pop()
#     # print (tfidf_matrix)
#     a = list(cosine_similarity(tfidf_matrix[-1:], tfidf_matrix))
#     b = list(a[0])
#     b.pop()
#     print (labels[b.index(max(b))])
