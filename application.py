from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the model with preprocessing pipeline
# Load the model and print the type
try:
    model = pickle.load(open('GredientBoostingRegressionModel.pkl', 'rb'))
    print("Model loaded successfully.")
    print("Loaded model type:", type(model))  # Check model type
    if hasattr(model, 'named_steps'):
        print("Pipeline steps:", model.named_steps)  # Check pipeline steps
except Exception as e:
    print("Error loading model:", e)


try:
    car = pd.read_csv('Car_price_Cleaned.csv')
    print("Car data loaded successfully.")
except Exception as e:
    print("Error loading car data:", e)

# Calculate 'car_age' based on 'manufacturing_year'
car['car_age'] = 2024 - car['manufacturing_year']

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['Company'].unique())
    car_model = sorted(car['car_name'].unique())
    car_age = sorted(car['car_age'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    transmission = car['transmission'].unique()
    ownership = car['ownership'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_model=car_model, car_age=car_age, fuel_types=fuel_type, transmission=transmission, ownership=ownership)

@app.route('/get_car_models/<company>', methods=['GET'])
@cross_origin()
def get_car_models(company):
    models = sorted(car[car['Company'] == company]['car_name'].unique())
    print("Models fetched for company:", company, "->", models)
    return {'models': models}

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Gather form inputs
    company = request.form.get('Company')
    car_model = request.form.get('car_name')
    car_age = request.form.get('car_age')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')
    ownership = request.form.get('ownership')
    transmission = request.form.get('transmission')

    # Log inputs for debugging
    print("Inputs:", company, car_model, car_age, driven, fuel_type, transmission, ownership)

    # Ensure all inputs are present
    if not all([company, car_model, car_age, driven, fuel_type, transmission, ownership]):
        return "All fields must be filled", 400  # Bad Request

    try:
        # Prepare the raw input data in the same format used for training
        input_data = pd.DataFrame(columns=['car_name', 'Company', 'car_age', 'kms_driven', 'fuel_type', 'transmission', 'ownership'],
                                  data=np.array([car_model, company, int(car_age), int(driven), fuel_type, transmission, ownership]).reshape(1, 7))
        print("Input DataFrame for prediction:\n", input_data)

        # Directly pass the raw input data to the pipeline
        prediction = model.predict(input_data)
        predicted_price = np.expm1(prediction)
        print("Prediction result:", predicted_price)

        return str(predicted_price)
    except Exception as e:
        print("Prediction Error:", e)
        return "Error in prediction: " + str(e), 500

            
# @app.route('/predict', methods=['POST'])
# @cross_origin()
# def predict():
#     company = request.form.get('Company')
#     car_model = request.form.get('car_name')
#     car_age = request.form.get('car_age')
#     fuel_type = request.form.get('fuel_type')
#     driven = request.form.get('kilo_driven')
#     ownership = request.form.get('ownership')
#     transmission = request.form.get('transmission')

#     print("Inputs:", company, car_model, car_age, driven, fuel_type, transmission, ownership)

#     if not all([company, car_model, car_age, driven, fuel_type, transmission, ownership]):
#         return "All fields must be filled", 400  # Bad Request

#     try:
#         # Prepare input as a DataFrame
#         input_data = pd.DataFrame({
#             'car_name': [car_model],
#             'Company': [company],
#             'car_age': [int(car_age)],
#             'kms_driven': [int(driven)],
#             'fuel_type': [fuel_type],
#             'transmission': [transmission],
#             'ownership': [ownership]
#         })
        
#         print("Input DataFrame for prediction:\n", input_data)

#         # Directly pass the raw input data to the pipeline
#         prediction = model.predict(input_data)
#         print("Prediction result:", prediction)

#         return str(np.round(prediction[0], 2))

#     except Exception as e:
#         print("Prediction Error:", e)
#         return "Error in prediction: " + str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
