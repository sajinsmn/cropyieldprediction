from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)

# loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
lgm = pickle.load(open('lgm.pkl', 'rb'))
xg = pickle.load(open('xg.pkl', 'rb'))
rf = pickle.load(open('rf.pkl', 'rb'))
knn = pickle.load(open('knn.pkl', 'rb'))
preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))

# flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']
        Model = request.form['Model']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                            dtype=object)
        transformed_features = preprocesser.transform(features)

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


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port)
