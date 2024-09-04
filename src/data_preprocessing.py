import pandas as pd
import numpy as np
import os

filepath = 'data/Raw/Dataset.csv'
def load_dataset(filepath=filepath):
    """
    Load a dataset from a CSV file.
    
    :param filepath: str, path to the dataset file
    :return: pd.DataFrame, loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    df = pd.read_csv(filepath)
    # if df has column 'Total' drop it =filepath
    if 'Total' in df.columns:
        df.drop(columns=['Total'], inplace=True)
    return df
    

def save_dataset(df, filepath):
    """
    Save a DataFrame to a CSV file.
    
    :param df: pd.DataFrame, the DataFrame to save
    :param filepath: str, path to the file where the DataFrame should be saved
    """
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")

def fill_missing_values(df):
    """
    Fill missing values in the dataset. Fill cost-related columns with 0 and 
    'Investment' with a default value if missing.
    
    :param df: pd.DataFrame, input dataset
    :return: pd.DataFrame, dataset with missing values filled
    """
    # Fill missing values in cost-related columns with 0
    cost_columns = ['Land_Planting', 'Strate_Fertilizer', 'Liquid_Fertilizer', 
                    'Fungicide', 'Insecticide', 'Others']
    for col in cost_columns:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Fill missing 'Investment' with a default value (e.g., 0)
    if 'Investment' not in df.columns:
        df['Investment'] = 0
    
    return df

def calculate_total_expenses(df):
    """
    Calculate the total expenses by summing all relevant cost columns.
    
    :param df: pd.DataFrame, input dataset containing cost columns
    :return: pd.DataFrame, dataset with the 'Total_Expenses' column added
    """
    cost_columns = ['Land_Planting', 'Strate_Fertilizer', 'Liquid_Fertilizer', 
                    'Fungicide', 'Insecticide', 'Others']
    if all(col in df.columns for col in cost_columns):
        df['Total_Expenses'] = df[cost_columns].sum(axis=1)
    return df

def calculate_yield_and_profitability(df, investment_value, price_per_kg, initial_cost=0):
    """
    Calculate the yield and profitability based on the investment value.
    
    :param df: pd.DataFrame, dataset with cost and yield data
    :param investment_value: float, amount of money available for expenses
    :param price_per_kg: float, price per KG of produce
    :param initial_cost: float, initial cost that affects the analysis (default is 0)
    :return: pd.DataFrame, dataset with calculated profitability
    """
    df['Investment'] = investment_value + initial_cost
    df['Estimated_Yield'] = (df['Investment'] / df['Total_Expenses']) * df['KG']
    df['Revenue'] = df['Estimated_Yield'] * price_per_kg
    df['Profit'] = df['Revenue'] - df['Investment']
    df['Is_Profitable'] = df['Profit'] > 0
    df['Cost_Per_Acre'] = df['Total_Expenses'] / df['Acre']  # Example feature: cost per acre
    df['Yield_Per_Acre'] =df['KG'] / df['Acre']  # Yield per acre
    return df

def preprocess_data(filepath=filepath, investment_value=0, price_per_kg=0, initial_cost=0, area=None, acres=0):
    """
    Complete preprocessing pipeline for the dataset.
    
    :param filepath: str, path to the dataset file
    :param investment_value: float, amount of money available for expenses
    :param price_per_kg: float, price per KG of produce
    :param initial_cost: float, initial cost that affects the analysis (default is 0)
    :return: pd.DataFrame, preprocessed dataset
    """
    df = load_dataset(filepath)
    df = fill_missing_values(df)
    df = calculate_total_expenses(df)
    df = calculate_yield_and_profitability(df, investment_value, price_per_kg, initial_cost)
    if area is not None and acres != 0:
        # Remove if colummns have unwanted spaces
        df = df.rename(columns=lambda x: x.strip())
        df = filter_data_by_area_and_acres(df, area, acres)        
    return df

def filter_data_by_area_and_acres(df, area, acres):
    """
    filter the dataset by the area and acres.
    if acres is 0, only filter by area.
    and also after it check it is empty or not.
    if empty create and return only with filtered area.
    based on acres and area.
    """
    if acres == 0:
        return filter_by_area(df, area)
    if df[(df['Area'] == area) & (df['Acre'] == acres)].empty:
        return filter_by_area(df, area)
    else:
        df =  df[(df['Area'] == area) & (df['Acre'] == acres)]
        return df    
    

def filter_by_area(df, area):
    """
    Filter the dataset by the area/location.
    
    :param df: pd.DataFrame, the dataset to filter
    :param area: str, the area to filter by
    :return: pd.DataFrame, filtered dataset
    """    
    return df[df['Area'] == area]

if __name__ == "__main__":
    # Example usage
    filepath = '../data/Raw/Dataset.csv'
    investment_value = 60000  # Example investment value
    price_per_kg = 100  # Example price per KG
    initial_cost = 0  # Example initial cost
    area = 'ampara'  # Example area to filter
    acres = 2  # Example
    
    processed_df = preprocess_data(filepath, investment_value, price_per_kg, initial_cost, area)
    if area and acres is not None:
        save_dataset(processed_df, f'../data/Processed/Preprocessed_Data_{area}.csv')
    else:
        save_dataset(processed_df, f'../data/Processed/Preprocessed_Data.csv')  
    