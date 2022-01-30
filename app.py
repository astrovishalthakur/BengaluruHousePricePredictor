from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", "rb"))


@app.route('/')
def index():
    locations = sorted(data.location.unique())
    return render_template('index.html', locations=locations)


@app.route("/predict", methods=["POST"])
def predict():

    location = request.form.get("location")
    bhk = float(request.form.get("bhk"))
    bath = float(request.form.get("bath"))
    sqft = float(request.form.get("total_sqft"))
    print(location, bhk, sqft, bath)

    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "BHK"])
    prediction = pipe.predict(input_data)
    print(prediction)
    return str(np.round(prediction[0] * 1e5, 2))


if __name__ == '__main__':
    app.run(debug=True)
