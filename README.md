# Spam Message Classifier Web App.

This repository contains all the code files develop for this web app project.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRBl5k6_UnrUOTX9idW8n52SaYiF4lZ4UflB4GmJ5AFDR3Uf19l">

This is a Machine Learning project which is deployed as a web application.

Preview: https://message-classifier.herokuapp.com

Objective: This project is to develop a machine learning model which can be used to predict wheather a recieved message is spam or not spam.

Programming language: Python3

Libraries: Sklearn, Pandas, NumPy etc.
Algorithm: Naive Bayes Classification Algorithm.

Web framework: Flask web framework in Python

Deployment: Heroku

skills used: programming, data analysis, machine learning algorithm knowledge, data structures and algorithm knowledge, web development.

Files introduction for this repository,

  1."spam.csv" is the dataset taken from UCI Machine Learning Repository.
  2."model.py" is the python script in which Machine Learning Model is developed.
  3."cv_vocabulary.py" and "NB_spam_model.py" is saved ML model after training it on the available dataset.
  4."app.py" has all the code used in developing web application for hosting this Machine Model as a service. This is developed using Flask     web framework.
  5."static" and "templates" folder consists of html and css files used in developing the web app.
  6.Finally deployed on Heroku, link: https://message-classifier.herokuapp.com
