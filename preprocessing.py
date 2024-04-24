import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def get_problem_type(df, target_column, classification_threshold=10):
    """
    Determine the type of the machine learning problem based on the target column.
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    - classification_threshold (int): Threshold for number of unique values to classify as classification problem.
    
    Returns:
    - str: Type of the machine learning problem ("regression" or "classification").
    """
    
    # Check the data type of the target column
    target_dtype = df[target_column].dtype

    # Check the number of unique values in the target column
    num_unique_values = df[target_column].nunique()

    # If the target column is numerical but has fewer unique values than the classification threshold, classify as classification problem
    if target_dtype in ['int64', 'float64'] and num_unique_values > classification_threshold:
        return "Regression"
    # If the target column is numerical, it's a regression problem
    elif num_unique_values == 2:
        return "Classification (Binary)"
    # If the target column is categorical, it's a classification problem
    elif num_unique_values > 2:
        return "Classification (Multiclass)" 
    # If the data type is not recognized, return None
    else:
        return None
        

def clean_data(df, target_column):
    """
    Clean the input DataFrame by handling missing values, removing duplicates, and encoding categorical variables.
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    
    Returns:
    - DataFrame: Cleaned DataFrame.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    df = df.loc[df[target_column].notna()]
    # Fill missing values
    # For numerical columns, use mean
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    
    # For categorical columns, use mode (most frequent value)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, axis=1)
    
    return df
