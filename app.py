#import Flask
#render_template for html pages
#url_for to get link for html pages
from flask import Flask, render_template, url_for, request

import pandas as pd 
import pickle

from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	cv_Vocabulary = open('cv_Vocabulary.pkl', 'rb')
	cv = joblib.load(cv_Vocabulary)

	NB_spam_model = open('NB_spam_model.pkl', 'rb')
	model = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]

		vector = cv.transform(data).toarray()

		prediction = model.predict(vector)

	return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
	app.run()
