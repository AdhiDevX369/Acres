from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sys
import os

# Import utility functions from your utils module
sys.path.append('src')
import utils as ut
import data_preprocessing as dp

app = Flask(__name__)

# Load the models
regression_model = joblib.load('models/regression_model.pkl')  # Path to your regression model
classification_model = joblib.load('models/classification_model.pkl')  # Path to your classification model

# Load the dataset to get the expected structure for one-hot encoding
df = pd.read_csv('data/Processed/Data_for_model.csv')

# Initialize the OneHotEncoder with the expected categories for the 'Area' column
encoder = OneHotEncoder(categories=[df['Area'].unique()], sparse_output=False, handle_unknown='ignore')

def preprocess_input(data):
    """
    Preprocess the input data by one-hot encoding categorical features and aligning columns.
    
    :param data: dict, input data
    :return: pd.DataFrame, preprocessed data
    """
    # Path for dataset
    dataset_path = 'data/Raw/Dataset.csv'
    
    # Load and preprocess the dataset
    df_input_og = dp.load_dataset(dataset_path)
    df_input = dp.preprocess_data(dataset_path, investment_value=data['Investment'], price_per_kg=data['Price_per_KG'], initial_cost=data['Initial_Cost'], area=data['Area'], acres=data['Acre'])

    # Ensure the encoder is fitted only once
    if not hasattr(encoder, 'categories_'):
        # Fit the encoder if it's not fitted yet
        encoder.fit(df[['Area']])
    
    # OneHotEncode the 'Area' column
    area_encoded = encoder.transform(df_input[['Area']])
    area_encoded_df = pd.DataFrame(area_encoded, columns=encoder.get_feature_names_out(['Area']))

    # Drop the original 'Area' column and concatenate the one-hot encoded columns
    df_input = df_input.drop('Area', axis=1)
    df_input_encoded = pd.concat([df_input, area_encoded_df], axis=1)

    # Align the columns of df_input_encoded with the original DataFrame used for training
    missing_cols = set(df.columns) - set(df_input_encoded.columns)
    for col in missing_cols:
        df_input_encoded[col] = 0

    # Ensure 'Target_Variable' is not included in the columns to be aligned
    columns_to_align = df.columns.drop('Target_Variable') if 'Target_Variable' in df.columns else df.columns
    df_input_encoded = df_input_encoded[columns_to_align]

    return df_input_encoded

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract data from the request
        investment_value = data.get('Investment_Value')
        number_of_acres = data.get('Acres')
        area = data.get('Area')
        price_per_kg = data.get('Price_per_KG')
        initial_cost = data.get('Initial_Cost', 0)  # Default to 0 if not provided

        # Ensure necessary data is provided
        if not all([investment_value, number_of_acres, area, price_per_kg]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Prepare data for predictions
        data_for_prediction = {
            'Investment': investment_value,
            'Acre': number_of_acres,
            'Area': area,
            'Price_per_KG': price_per_kg,
            'Initial_Cost': initial_cost
        }
        
        # Convert input data to DataFrame and preprocess
        features = preprocess_input(data_for_prediction)
        
        # Predict using the regression model (e.g., yield prediction)
        yield_prediction = regression_model.predict(features)[0]

        # Predict using the classification model (e.g., profitability)
        profitability_prediction = classification_model.predict(features)[0]

        # Calculate adjusted profitability
        adjusted_profitability = yield_prediction * price_per_kg - investment_value - initial_cost
        is_profitable_adjusted = adjusted_profitability > 0

        # Prepare the response
        results = {
            'Predicted_Yield': yield_prediction,
            'Profitability_Prediction': bool(profitability_prediction),
            'Adjusted_Profitability': adjusted_profitability,
            'Is_Profitable_Adjusted': is_profitable_adjusted
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)