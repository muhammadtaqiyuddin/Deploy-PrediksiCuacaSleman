from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__, static_folder='images')

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Get the model's feature names
X_columns = model.feature_names_in_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['temperature'])
        wind_speed = float(request.form['wind_speed'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        radiation = float(request.form['radiation'])

        input_data = {
            'Tn': temperature,
            'ff_avg': wind_speed,
            'RH_avg': humidity,
            'RR': rainfall,
            'ss': radiation,
            'ddd_car_E': 0,
            'ddd_car_N': 0,
            'ddd_car_NE': 0,
            'ddd_car_NW': 0,
            'ddd_car_S': 0,
            'ddd_car_SE': 0,
            'ddd_car_SW': 0,
            'ddd_car_W': 0,
            'Tx': 0  # Adding this feature assuming it was in the trained model
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.rename(columns=lambda x: x.strip())

        # Add missing columns with default values
        for column in X_columns:
            if column not in input_df:
                input_df[column] = 0

        input_df = input_df[X_columns]

        prediction = model.predict(input_df)
        return jsonify(prediction=prediction[0])
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True)
