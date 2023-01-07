# -*- coding: utf-8 -*-
import re
from flask import Flask,render_template,url_for,request
import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()
from nltk.corpus import stopwords

# load the model from disk
clf = pickle.load(open('xgb.pkl', 'rb'))
cv=pickle.load(open('tfidf.pkl','rb'))
app = Flask(__name__)
def preprocess(text):
    
    # Convert text to lowercase
    text = text.apply(lambda x: x.lower())
    
    # Remove special characters and digits
    text = text.apply(lambda x: re.sub(r'([^A-Za-z|\s|[:punct:]]*)', '', x))
    
    # Replace certain characters and words with spaces
    text = text.apply(lambda x: x.replace('[^a-zA-Z#]', ' ').replace('quot', '').replace(':', '').replace('sxsw', ''))
    
    # Remove words that are shorter than 2 characters
    text = text.apply(lambda x: ' '.join([i for i in x.split() if len(i) > 1]))
    
    # Tokenize the text
    text = text.apply(lambda x: x.split())
    
    # Lemmatize the tokens
    text = text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    
    # Remove stopwords
    text = text.apply(lambda x: [word for word in x if word not in stopwords])
    
    # Join the preprocessed tokens back into a single string
    text = text.apply(lambda x: ' '.join(x))
    
    return text

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if(len(message)>2):
            text = [message]
            data = preprocess(text)
            vect = cv.transform(data)
            my_prediction = clf.predict(vect)
        else:
            my_prediction=3
        
    return render_template('home.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
