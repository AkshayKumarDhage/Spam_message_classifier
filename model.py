#import required libraries
import pandas as pd 
import numpy as np 

#CountVectorizer(): Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

#seperate data into test and train set
from sklearn.model_selection import train_test_split

#MultinomialNB(): Naive Bayes Classifier for multinomial models
#Multinomial Naive Bayes classifier is suitable for classification with discrete features(eg. word counts for text classification)
from sklearn.naive_bayes import MultinomialNB

#classification report for the trained model and statistics
from sklearn.metrics import classification_report


#read the spam.csv dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

#dataset has features = [class, message, Unnamed: 2, Unnamed: 3, Unnamed: 4]
#we shall drop all Unnamed features
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

#label 0 for ham and 1 for spam
#feature 'class' can have two values 'spam' and 'ham'
#categorical values to numerical values
df['label'] = df['class'].map({'ham': 0, 'spam': 1})

#seperate dependent and Independent/target features
X = df['message']
y = df['label']

#CountVectorizer()
cv = CountVectorizer()

X = cv.fit_transform(X)

#split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes model
model = MultinomialNB()

#fit model on train set
model.fit(X_train, y_train)

model.score(X_test, y_test)

#prediction for test set
y_pred = model.predict(X_test)

#classification report
#print(classification_report(y_test, y_pred))

#saving the model for future use
from sklearn.externals import joblib
joblib.dump(model, 'NB_spam_model.pkl')
joblib.dump(cv, 'cv_Vocabulary.pkl')
