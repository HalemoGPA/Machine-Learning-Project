from flask import Flask, request, jsonify
import pandas as pd
import pickle
# from preprocessing import HandleSmokingStatus

app = Flask(__name__)

# Load preprocessing pipeline
with open('preprocessing_pipeline.pkl', 'rb') as file:
    preprocessing_pipeline = pickle.load(file)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# API endpoint for making predictions
@app.route('/api', methods=["GET"])
def predict():
    # Get data from request
    req_data = {
        'gender': request.args['gender'],
        'age': float(request.args['age']),
        'hypertension': int(request.args['hypertension']),
        'heart_disease': int(request.args['heart_disease']),
        'ever_married': request.args['ever_married'],
        'work_type': request.args['work_type'],
        'Residence_type': request.args['Residence_type'],
        'avg_glucose_level': float(request.args['avg_glucose_level']),
        'bmi': float(request.args['bmi']),
        'smoking_status': request.args['smoking_status']
    }
    
    # Transform the input data using the preprocessing pipeline
    data = pd.DataFrame([req_data])
    transformed_data = preprocessing_pipeline.transform(data)

    # Make predictions using the model
    predictions = model.predict(transformed_data)

    # Return predictions as JSON response
    if predictions == 0:
        return jsonify({"output": "not stroke"}), 200
    else:
        return jsonify({"output": "stroke"}), 200

if __name__ == '__main__':
    app.run(debug=True)
