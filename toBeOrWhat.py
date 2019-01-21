# Enter your code here. Read input from STDIN. Print output to STDOUT

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import TfidfTransformer



def cleanData(data):
    # data[i] = re.sub("[^a-zA-Z]"," ", data[i])
    data = data.lower()
    data  = re.sub(r"\\", "", data )    
    data  = re.sub(r"\'", "", data )    
    data  = re.sub(r"\"", "", data )    
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    data = data.translate(translate_map)
    return data

def getTrainText(raw):
  data = []
  labels = []
  labelsGIven = ['am','are','were','was','is','been','being','be']
  for x in raw.split("."):
    x = cleanData(x)
    if "?" in x :
        y = x.split("?")
        for strings in y :
            for word in labelsGIven:
                if word in strings:
                    strings.replace(word," ")
                    if re.search('[a-zA-Z]', strings):
                        data.append(strings)
                        labels.append(word)
    else:
        for word in labelsGIven :
            if word in x:
                x.replace(word," ")
                if re.search('[a-zA-Z]', x):
                    data.append(x)
                    labels.append(word)
  return data ,labels            

with open('./corpus.txt', 'r') as content_file:
    raw = content_file.read()

# print (raw)
data,labels = getTrainText(raw)
# print (len(data) , len(labels))

# getting the input data
n = int (input())
testText = input()

testData = []
def countFreq(pat, txt): 
    M = len(pat) 
    N = len(txt) 
    res = 0
      
    # A loop to slide pat[] one by one  
    for i in range(N - M + 1): 
          
        # For current index i, check  
        # for pattern match  
        j = 0
        for j in range(M): 
            if (txt[i + j] != pat[j]): 
                break
  
        if (j == M - 1): 
            res += 1
            j = 0
    return res 

for x in testText.split("."):
    if "----" in x:
        # for y in 
        m = countFreq("----", x)
        x.replace("----","")
        x = cleanData(x)
        # if re.search('[a-zA-Z]', x):
        # print (m)
        for l in range(int(m/2)):
            testData.append(x)


# print (testData)
data, labels = shuffle(data, labels, random_state=0)


count_vect = CountVectorizer(ngram_range = (1,1),max_df = 0.1)
tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_counts = count_vect.fit_transform(data[:1050])
# X_train_counts2 = count_vect.transform(data[5000:])
testData = count_vect.transform(testData)


trainData = tfidf_transformer.fit_transform(X_train_counts)

testData  = tfidf_transformer.transform(testData)

clf = RandomForestClassifier(n_estimators=100,max_depth = None,random_state = 0)
clf.fit(trainData.toarray(), labels[:1050])
ans = clf.predict(testData.toarray())
for x in ans:
    print (x)
