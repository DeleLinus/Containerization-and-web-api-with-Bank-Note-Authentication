from flask import Flask, request
import numpy as np
import pandas as pd
import pickle


app=Flask(__name__)
with open("model.pkl","rb") as pickle_in:
	classifier = pickle.load(pickle_in)

#make a constructor using @name.route("/"). "/" specifies the first page, same 
#as the initial url
@app.route("/")
def welcome():
    return "You are welcome"

#route method without a method is initially aset to GET
@app.route("/predict")
def predict_note_authentication():
    #the argument passed inside the fet method are used as param KEY in postman
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is " + str(prediction)

@app.route("/predict_file", methods=["POST"])
def predict_from_file():
    df_file = pd.read_csv(request.file.get("file"))
    prediction = classifier.predict(df_file)
    
    return "The predicted values from the file are " + str(list(prediction))
    



if __name__ == "__main__":
	app.run()