import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def Preprocess_Dropna(data, allowNormalized = False):
    # Check for missing values
    print("Missing Values:")
    print(data.isnull().sum())

    # Drop rows with missing values 
    data = data.dropna()

    if allowNormalized == True:
        return Normalize_Data(data)
    else:
        return data

def Preprocess_Fillna(data, allowNormalized = False):
    data['Date'] = pd.to_datetime(data['Date'])
    # Sort the data by date in ascending order
    data = data.sort_values('Date')

    # Check for missing values
    print("Missing Values:")
    print(data.isnull().sum())

    # Forward fill missing values
    # Other filling methods such as backfill, mean, median, mode can also be used.
    data.fillna(method='ffill', inplace=True)

    if allowNormalized == True:
        return Normalize_Data(data)
    else:
        return data

def Normalize_Data(data):
    # Separate the 'Date' column from the numeric columns
    numeric_columns = [col for col in data.columns if col != 'Date']
    
    # Convert the numeric values to numeric format
    numeric_data = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values in the numeric data
    numeric_data = numeric_data.dropna()

    # Normalize the numeric data
    mms = MinMaxScaler()
    normalized_data = mms.fit_transform(numeric_data)
    
    # Create a new DataFrame with the normalized values
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)
    
    # Combine with the 'Date' column
    processed_data = pd.concat([data['Date'], normalized_df], axis=1)
    
    # Return the normalized data
    return processed_data