from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
import os
import requests

print("✅ sklearn version:", sklearn.__version__)

# ==============================
# Direct Google Drive Download
# ==============================
# Replace this with your direct download link
RF_FILE_URL = "https://drive.google.com/uc?export=download&id=1IlSdQ_BenMyelaNYkXrZ8EgQjyLcVHQH"
RF_FILE_PATH = "rf.pkl"

# Download rf.pkl if not exists
if not os.path.exists(RF_FILE_PATH):
    print("📥 Downloading rf.pkl from Google Drive...")
    r = requests.get(RF_FILE_URL, allow_redirects=True)
    with open(RF_FILE_PATH, "wb") as f:
        f.write(r.content)

# ==============================
# Load Models
# ==============================
dtr = pickle.load(open('dtr.pkl', 'rb'))
lgm = pickle.load(open('lgm.pkl', 'rb'))
xg = pickle.load(open('xg.pkl', 'rb'))
rf = pickle.load(open(RF_FILE_PATH, 'rb'))  # downloaded model
knn = pickle.load(open('knn.pkl', 'rb'))
preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    Year = request.form['Year']
    average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
    pesticides_tonnes = request.form['pesticides_tonnes']
    avg_temp = request.form['avg_temp']
    Area = request.form['Area']
    Item = request.form['Item']
    Model = request.form['Model']

    # preprocess
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                        dtype=object)
    transformed_features = preprocesser.transform(features)

    # choose model
    if Model == 'dtr':
        prediction = dtr.predict(transformed_features).reshape(1, -1)
    elif Model == 'knn':
        prediction = knn.predict(transformed_features).reshape(1, -1)
    elif Model == 'lgm':
        prediction = lgm.predict(transformed_features).reshape(1, -1)
    elif Model == 'xg':
        prediction = xg.predict(transformed_features).reshape(1, -1)
    elif Model == 'rf':
        prediction = rf.predict(transformed_features).reshape(1, -1)

    return render_template('index.html', prediction=prediction[0][0])

# ==============================
# Render Deployment
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT dynamically
    app.run(host="0.0.0.0", port=port)
