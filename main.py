import pandas as pd
from EDA import Perform_EDA
from ML import Modeling_RandomForestClassifier
from ML import Modeling_XGBoost
from DataProcessing import Preprocess_Dropna
from DataProcessing import Preprocess_Fillna

data = pd.read_csv(r'C:\Users\beyza.uzeyiroglu\Desktop\data.csv')

# Drop missing values 
dropna_data = Preprocess_Dropna(data, True); 
# Exploratory data analysis
Perform_EDA(dropna_data)

# Complete missing values
imputed_data = Preprocess_Fillna(data, True);
# Exploratory data analysis
Perform_EDA(imputed_data)

# Must be taken from outside(maybe with integration)
new_data = pd.DataFrame({
    'Open': [123.45],
    'High': [127.89],
    'Low': [121.34],
    'Close': [126.78],
    'Adj Close': [125.50],
    'Volume': [1000000]
})

# Models with dropna data
Modeling_RandomForestClassifier(dropna_data, new_data)
Modeling_XGBoost(dropna_data, new_data)

# Models with imputed data
Modeling_RandomForestClassifier(imputed_data, new_data)
Modeling_XGBoost(imputed_data, new_data)

