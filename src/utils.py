import pandas as pd
import numpy as np
import os
import joblib 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_dataset(filepath):
    """
    Load a dataset from a CSV file.
    
    :param filepath: str, path to the dataset file
    :return: pd.DataFrame, loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_csv(filepath)

def save_dataset(df, filepath):
    """
    Save a DataFrame to a CSV file.
    
    :param df: pd.DataFrame, the DataFrame to save
    :param filepath: str, path to the file where the DataFrame should be saved
    """
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")

def calculate_total_expenses(df):
    """
    Calculate the total expenses by summing all relevant cost columns.
    
    :param df: pd.DataFrame, input dataset containing cost columns
    :return: pd.DataFrame, dataset with the 'Total Expenses' column added
    """
    cost_columns = ['Land_Planting', 'Strate_Fertilizer', 'Liquid_Fertilizer', 
                    'Fungicide', 'Insecticide', 'Others']
    df['Total_Expenses'] = df[cost_columns].sum(axis=1)
    return df

def filter_by_area(df, area):
    """
    Filter the dataset by the area/location.
    
    :param df: pd.DataFrame, the dataset to filter
    :param area: str, the area to filter by
    :return: pd.DataFrame, filtered dataset
    """    
    return df[df['Area'] == area]

def visualize_yield_vs_expense(df):
    """
    Create a simple scatter plot of Total Expenses vs. Total Yield.
    
    :param df: pd.DataFrame, the dataset to plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.scatterplot(x='Total_Expenses', y='KG', data=df)
    plt.title('Total Expenses vs. Total Yield')
    plt.xlabel('Total Expenses')
    plt.ylabel('Total Yield (KG)')
    plt.show()
    
def fixed_columns(df):
    """    Fix column names by removing leading/trailing spaces.    
    """
    df = df.rename(columns=lambda x: x.strip())
    return df


# Calculate yield and profitability based on investment value
def calculate_yield_and_profitability(investment_value, df, price_per_kg, initial_cost=0):
    """
    Calculate the yield and profitability based on the investment value.
    
    :param investment_value: float, amount of money available for expenses
    :param df: pandas DataFrame with cost and yield data
    :param price_per_kg: float, price per KG of produce
    :param initial_cost: float, initial cost that affects the analysis (default is 0)
    :return: pandas DataFrame with calculated profitability
    """
    df['Investment'] = investment_value + initial_cost
    df['Estimated_Yield'] = (investment_value / df['Total_Expenses']) * df['KG']
    df['Revenue'] = df['Estimated_Yield'] * price_per_kg
    df['Profit'] = df['Revenue'] - df['Investment']
    df['Is_Profitable'] = df['Profit'] > 0
    return df

# Summarize investment analysis by area
def summarize_investment_analysis(df):
    """
    Summarize the investment analysis by area, calculating average profitability.
    
    :param df: pandas DataFrame
    :return: pandas DataFrame with summarized data
    """
    summary = df.groupby('Area').agg({
        'Profit': 'mean',
        'Is_Profitable': 'mean',
        'Estimated_Yield': 'mean'
    }).reset_index()
    return summary

# Train and save the model
def train_and_save_model(model, X_train, y_train, model_path='yield_prediction_model.pkl'):
    """
    Train the model and save it to a file.
    
    :param model: machine learning model to train
    :param X_train: features for training
    :param y_train: target for training
    :param model_path: str, path to save the trained model (default is 'yield_prediction_model.pkl')
    :return: trained model
    """
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using common metrics.
    
    :param model: trained machine learning model
    :param X_test: features for testing
    :param y_test: target for testing
    :return: dict with evaluation metrics (MAE, MSE, R-squared)
    """
    y_pred = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R-squared': r2_score(y_test, y_pred)
    }
    return metrics

# Load the trained model
def load_model(model_path='yield_prediction_model.pkl'):
    """
    Load a trained model from a file.
    
    :param model_path: str, path to the trained model file (default is 'yield_prediction_model.pkl')
    :return: loaded model
    """
    return joblib.load(model_path)

def one_hot_encode(df, column):
    """
    Perform One-Hot Encoding on the specified column.
    
    :param df: pandas DataFrame
    :param column: str, column to encode
    :return: pandas DataFrame with one-hot encoded column
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated parameter
    encoded_columns = pd.DataFrame(encoder.fit_transform(df[[column]]), 
                                   columns=encoder.get_feature_names_out([column]))
    df = df.drop(column, axis=1)
    df = pd.concat([df, encoded_columns], axis=1)
    return df

# Label Encoding
def label_encode(df, column):
    """
    Perform Label Encoding on the specified column.
    
    :param df: pandas DataFrame
    :param column: str, column to encode
    :return: pandas DataFrame with label encoded column
    """
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df

# Target Encoding
def target_encode(df, column, target):
    """
    Perform Target Encoding on the specified column.
    
    :param df: pandas DataFrame
    :param column: str, column to encode
    :param target: str, target column for encoding
    :return: pandas DataFrame with target encoded column
    """
    encoder = ce.TargetEncoder(cols=[column])
    df[column] = encoder.fit_transform(df[column], df[target])
    return df

# Evaluate encoding techniques
def evaluate_encoding_techniques(df, target, categorical_column):
    """
    Evaluate different encoding techniques for a categorical column.
    
    :param df: pandas DataFrame
    :param target: str, target column name
    :param categorical_column: str, categorical column to encode
    :return: dict with encoding techniques and their R-squared scores
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    techniques = {
        'One-Hot Encoding': one_hot_encode(df.copy(), categorical_column),
        'Label Encoding': label_encode(df.copy(), categorical_column),
        'Target Encoding': target_encode(df.copy(), categorical_column, target)
    }
    
    scores = {}
    
    for technique, encoded_df in techniques.items():
        model = LinearRegression()
        score = cross_val_score(model, encoded_df, y, cv=5, scoring='r2').mean()
        scores[technique] = score
    
    return scores

# Plot encoding technique performance
def plot_encoding_performance(scores):
    """
    Plot the performance of different encoding techniques.
    
    :param scores: dict with encoding techniques and their scores
    """
    plt.bar(scores.keys(), scores.values())
    plt.xlabel('Encoding Technique')
    plt.ylabel('R-squared Score')
    plt.title('Encoding Technique Performance')
    plt.show()

# Get the best encoding technique
def get_best_encoding_technique(scores):
    """
    Determine the best encoding technique based on R-squared scores.
    
    :param scores: dict with encoding techniques and their scores
    :return: str, name of the best encoding technique
    """
    best_technique = max(scores, key=scores.get)
    print(f"Best encoding technique: {best_technique} with R-squared score: {scores[best_technique]}")
    return best_technique