
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def Modeling_RandomForestClassifier(data, new_data):
    # Split the data into features and target
    features = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    target = (data['Close'].shift(-1) - data['Close']) > 0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    not_nan_indices = ~np.isnan(X_test_scaled).any(axis=1)
    X_test_scaled = X_test_scaled[not_nan_indices]
    y_test = y_test[not_nan_indices]

    # Drop rows with missing values
    not_nan_indices = ~np.isnan(X_train_scaled).any(axis=1)
    X_train_scaled = X_train_scaled[not_nan_indices]
    y_train = y_train[not_nan_indices]

    # Train the Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Prediction new data
    prediction = clf.predict(new_data)

    # Decide whether to buy
    if prediction[0]:
        return print("Buy stock.")
    else:
        return print("Don't buy stock.")



def Modeling_XGBoost(data, new_data):
    # Split the data into features and target
    features = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    target = (data['Close'].shift(-1) - data['Close']) > 0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Normalize the features
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the XGBoost Classifier
    clf = xgb.XGBClassifier()
    clf.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    prediction = clf.predict(new_data)

    if prediction[0]:
        print("Buy stock.")
    else:
        print("Don't buy stock.")