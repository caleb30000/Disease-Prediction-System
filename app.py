import pickle
import numpy as np
import pandas as pd
from database import *
from scipy.stats import mode
from flask import Flask, redirect, render_template, request, url_for

import warnings
warnings.filterwarnings("ignore")

session = {
    "user": "",
    "result_1": "",
    "result_2": "",
    "result_3": "",
    "result_4": "",
    "model_1": "Random Forest Classifier",
    "model_2": "Naive Bayes Classifier",
    "model_3": "SVM Classifier",
    "model_4": "Ensemble",
    "first_symptom": "",
    "second_symptom": "",
    "third_symptom": "",
    "fourth_symptom": "",
}


# Deserializing models and encoder
rf_model = pickle.load(open("RandomF-model.pkl", "rb"))
nb_model = pickle.load(open("GaussianNB-model.pkl", "rb"))
svm_model = pickle.load(open("SVM-model.pkl", "rb"))
encoder = pickle.load(open("prognosis-encoder.pkl", "rb"))

# Working on data encoding used to train model.
data = pd.read_csv('Disease Prediction Model/dataset/training.csv')
X = data.iloc[:,:-1]
symptoms = X.columns.values

# Creating a symptom index dictionary to encode the input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Creating Function for prediction
def predict_disease(symptoms):
    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    # Reshaping the input data and converting it into suitable format for model predictions
    # input_data = input_data # THIS IS NEW
    input_data = np.array(input_data[:132]).reshape(1,-1)
    # Generating individual outputs
    rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]], # data_dict["predictions_classes"][rf_model.predict_proba(input_data)[0]] 
    nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]], # data_dict["predictions_classes"][nb_model.predict_proba(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]], # data_dict["predictions_classes"][svm_model.predict_proba(input_data)[0]]
    # Making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0][0]
    # Storing data into session.
    session["result_1"] = rf_prediction[0]
    session["result_2"] = nb_prediction[0]
    session["result_3"] = svm_prediction[0]
    session["result_4"] = final_prediction
    return (rf_prediction[0], nb_prediction[0], svm_prediction[0], final_prediction)


app = Flask(__name__)

@app.route("/sign-up/", methods= ["GET", "POST"])
def sign_up():
    if request.method == "GET":
        return render_template("sign-up.html")
    else:
        full_name = request.form.get("name")
        email_address = request.form.get("email")
        password = request.form.get("password")
        button = request.form.get("button")

        if button == "register":
            # Storing user in database
            if db.search(User.email == email_address):
                return render_template("sign-up.html", message= "User already exists.")
            else:
                create_user(full_name, email_address, password)
                return render_template("sign-in.html", message= "User registered, Log in!")
        if button == "log-in":
            if (db.search(User.email == email_address) and db.search(User.password == password)):
                session["user"] = db.search(User.email == email_address)[0]["full_name"].title()
                return redirect(url_for("master_page"))
            else:
                return render_template("sign-up.html", message= "Invalid Username or Password, Sign up.")
 

@app.route("/master-page/", methods= ["GET", "POST"])
def master_page():
    if request.method == "GET":
        return render_template("first.html", user_ = session["user"])
    else:
        # Collection of symptoms from form.
        first_symptom = request.form.get("symp1").title()
        second_symptom = request.form.get("symp2").title()
        third_symptom = request.form.get("symp3").title()
        fourth_symptom = request.form.get("symp4").title()

        # Storing symptoms in session
        session["first_symptom"] = first_symptom
        session["second_symptom"] = second_symptom
        session["third_symptom"] = third_symptom
        session["fourth_symptom"] = fourth_symptom

        # Function to predict diseases.
        predict_disease([first_symptom, second_symptom, third_symptom, fourth_symptom])
        return redirect(url_for("final_page"))


@app.route("/final-page/", methods= ["GET", "POST"])
def final_page():
        return render_template(
            "final.html", user_= session["user"], model_1 = session["model_1"], result_1 = session["result_1"],
            model_2 = session["model_2"], result_2 = session["result_2"], model_3 = session["model_3"],
            result_3 = session["result_3"], model_4 = session["model_4"], result_4 = session["result_4"],
            symp_1 = session["first_symptom"], symp_2 = session["second_symptom"], symp_3 = session["third_symptom"],
            symp_4 = session["fourth_symptom"])


if __name__ == "__main__":
    app.run()