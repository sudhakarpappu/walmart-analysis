#Team ID=SWTID1720435231

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import pickle
import datetime as dt
import calendar
import os

app = Flask(__name__)

# Loading the model and dataset
loaded_model = pickle.load(open('project\models.pkl', 'rb'))
train_data = pd.read_csv('project/trains.csv')

@app.route('/')
def home():
    print("Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route accessed")
    # Collect form data
    store = request.form.get('store')
    dept = request.form.get('dept')
    date = request.form.get('date')
    isHoliday = request.form.get('isHolidayRadio', 'False')  # Default value if not selected
    size = request.form.get('size')
    temp = request.form.get('temp')

    # Convert date string to datetime object
    d = dt.datetime.strptime(date, '%Y-%m-%d')
    year = d.year
    month = d.month
    month_name = calendar.month_name[month]

    # Ensure input values are converted to appropriate types
    store = int(store)
    dept = int(dept) if dept else 0  # Handle NoneType for dept gracefully
    size = int(size)
    isHoliday = isHoliday == 'True'
    
    # Check if temp is None or empty string, set default value if necessary
    if temp is None or temp == '':
        temp = 0.0
    else:
        temp = float(temp)

    # Create the test DataFrame for prediction
    X_test = pd.DataFrame({
        'Store': [store],
        'Dept': [dept],
        'Size': [size],
        'IsHoliday_x': [int(isHoliday)],  # Assuming you used 'IsHoliday_x' during training
        'CPI': [212.0],  # Assuming CPI value
        'Temperature': [temp],
        'Type_B': [0],   # Assuming binary indicators
        'Type_C': [1],
        'month': [month],
        'year': [year],
        'is_weekend': [0]  # Assuming 'is_weekend' is 0 for simplicity
    })

    # Make a prediction using the loaded model
    y_pred = loaded_model.predict(X_test)
    output = round(y_pred[0], 2)
    print("Predicted = ", output)

    # Render template with results
    return render_template('index.html', output=output, store=store, dept=dept, month_name=month_name, year=year)


if __name__ == "__main__":
    port = int(os.getenv('VCAP_APP_PORT', '4000'))  # Set a default port if not provided
    print(f"Starting app on port {port}")
    app.run(host='localhost', port=port)
