from flask import Flask,render_template,request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import nltk

app = Flask(__name__)
Pkl_Filename = open('models/spam_detect.pkl','rb')  
Pickled_LR_Model = pickle.load(Pkl_Filename)

tokenizer = RegexpTokenizer('\w+')
sw = set(stopwords.words('english'))
ps = PorterStemmer()
le = LabelEncoder()

def getStem(review):
    review = review.lower()
    tokens = tokenizer.tokenize(review) # breaking into small words
    removed_stopwords = [w for w in tokens if w not in sw]
    stemmed_words = [ps.stem(token) for token in removed_stopwords]
    clean_review = ' '.join(stemmed_words)
    return clean_review

def getDoc(document):
    d = []
    print("Byee")
    for doc in document:
        d.append(getStem(doc))
    return d

cv = pickle.load(open("models/vectorizer.pickle", 'rb'))

def prepare(messages):
    print("hello")
    d = getDoc(messages)
    # dont do fit_transform!! it will create new vocab.
    return cv.transform(d)

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        message = request.form['msg']
        messages = []
        messages.append(message)
        messages = prepare(messages)
        y_pred = Pickled_LR_Model.predict(messages)
        return render_template('show.html',hspam=y_pred[0].upper(),message=message)
    return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True,port=7000)