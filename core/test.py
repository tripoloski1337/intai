# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import glob
import time
import pandas as pd
# from xml.dom import minidom
from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd

# for neural network
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

def Train():

    # from sklearn.feature_extraction.text import TfidfVectorizer
    # def makeTokens(f):
    #     tkns_BySlash = str(f.encode('utf-8')).split('/')
    #     total_Tokens = []
    #     for i in tkns_BySlash:
    #         tokens = str(i).split('-')
    #         tkns_ByDot = []
    #         for j in range(0, len(tokens)):
    #             temp_Tokens = str(tokens[j]).split('.')
    #             tkns_ByDot = tkns_ByDot + temp_Tokens
    #         total_Tokens = total_Tokens + tokens + tkns_ByDot
    #     total_Tokens = list(set(total_Tokens))
    #     if 'com' in total_Tokens:
    #         total_Tokens.remove('com')
    #     return total_Tokens

    df = pd.read_csv("./sqli.csv",encoding='utf-16')
    vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
    posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()
    transformed_posts=pd.DataFrame(posts)
    df=pd.concat([df,transformed_posts],axis=1)
    X=df[df.columns[2:]]
    y=df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    # print("{} accuracy".format(clf.score(X_test, y_test)))

    y_pred = clf.predict(X_test)
    print("accuracy score: {}".format(accuracy_score(y_test, y_pred)))

    x_predict = ["'UNION SELECT * FROM "] # ini data yang bakal di deteksi
    x_predict = vectorizer.transform(x_predict)
    New_predict = clf.predict(x_predict)
    print(New_predict)


    # using simple neural network
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(layers.Dense(20, input_dim=input_dim, activation="relu"))
    model.add(layers.Dense(10, activation='tanh'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    model.summary()

    classifier_nn = model.fit(X_train, y_train, epochs=10, verbose=True, validation_data=(X_test, y_test), batch_size=15)
    model.save("./model/sqli.h5")
    print("saved trained data to ./model/sqli.h5")

    pred = model.predict(x_predict)
    print(pred) # [[0.99413866]]

def nn_loadModule():
    df = pd.read_csv("./dataset/sqli.csv",encoding='utf-16')
    vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
    posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()
    x_predict = ["' select * from information_scheme"] # ini data yang bakal di deteksi
    x_predict = vectorizer.transform(x_predict)
    print(x_predict)
    
    model = load_model('./model/sqli.h5')
    x = model.predict(x_predict)
    print("accuracy: {} ".format(x[0][0]))
    print(round(x[0][0])) # [[0.5007137]]



nn_loadModule()
# Train()