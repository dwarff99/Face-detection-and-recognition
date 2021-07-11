
#'from datetime import time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from flask import Flask, render_template, request
import webbrowser
import flask
import os
app = Flask(__name__)
dataset = pd.read_csv('cleanprojectdataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.4, random_state = 42)
X_train.shape
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf',SVC()),])
text_clf.fit(X_train.ravel(), y_train.ravel())
predictions = text_clf.predict(X_test.ravel())
from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

inputt = input("enter the text")
new = []
new.append(inputt)
# new = input("Enter the text")
#final=[np.array(new)]

n = text_clf.predict(new)
print("this sentence is :")
print(n)

pickle.dump(n,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

with open( 'model_savesvm', 'wb') as f:
    pickle.dump(text_clf, f)

with open('model_savesvm', 'rb') as f:
    b = pickle.load(f)
    c = b.predict(new)
    print("this sentence is 2:")
    print(c)
    print(metrics.accuracy_score(y_test, predictions))
