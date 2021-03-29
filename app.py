import os
import flask
import numpy as np
import pickle
from flask import Flask, render_template, request

app=Flask(__name__,static_url_path='/static')

#home page
@app.route("/")
def home():
 return render_template("index.html")


#form page
@ app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html')
   
 
def value_predictor(predict_list):
 to_predict = np.array(predict_list).reshape(1,7)
 loaded_model = pickle.load(open("RandomForest.pkl","rb"))
 result = loaded_model.predict(to_predict)
 return result[0]
 
@app.route("/predict",methods = ["POST"])
def result():
 if request.method == "POST":
    predict_list = request.form.to_dict()
    predict_list=list(predict_list.values())
    predict_list = list(map(float, predict_list))
    result = value_predictor(predict_list)
    prediction = str(result)
 return render_template("predict.html",prediction=prediction)
 
if __name__ == "__main__":
 app.run(debug=True)
 