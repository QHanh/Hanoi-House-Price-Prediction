import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

def prepare_input_data(df, categorical_features, all_columns):
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    df_encoded = df_encoded.reindex(columns=all_columns, fill_value=False)
    return df_encoded

df = pd.read_csv("../clean/data_ETLed.csv")
X = df[['Estate_type', 'District', 'Ward', 'Square', 'Numb_bedroom', 'Numb_toilet', 'Numb_floor']]
categorical_features = ['Estate_type', 'District', 'Ward']
X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)
X_numeric = X[['Square', 'Numb_bedroom', 'Numb_toilet', 'Numb_floor']]
X_preprocessed = pd.concat([X_numeric, X_categorical], axis=1)
all_columns = X_preprocessed.columns

model = pickle.load(open("../model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    estate_type = request.form['Estate_type']
    district = request.form['District']
    ward = request.form['Ward']
    square = float(request.form['Square'])
    numb_bedroom = int(request.form['Numb_bedroom'])
    numb_toilet = int(request.form['Numb_toilet'])
    numb_floor = int(request.form['Numb_floor'])

    input_data = pd.DataFrame({
        'Estate_type': [estate_type],
        'District': [district],
        'Ward': [ward],
        'Square': [square],
        'Numb_bedroom': [numb_bedroom],
        'Numb_toilet': [numb_toilet],
        'Numb_floor': [numb_floor]
    })

    input_preprocessed = prepare_input_data(input_data, ['Estate_type', 'District', 'Ward'], all_columns)
    
    prediction = model.predict(input_preprocessed)
    output = prediction[0]
    
    if output >= 1e9:
        formatted_output = f"{output / 1e9:.2f} tỷ"
    else:
        formatted_output = f"{output:.2f}"
    
    return render_template("index.html", prediction_text=f"Predicted Value: {formatted_output}")

if __name__ == "__main__":
    app.run(debug=True)



