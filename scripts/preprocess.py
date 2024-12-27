import os

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

ARTIFACTS_DIR = os.getenv("PROCESSED_DIR","/Users/aswathshakthi/PycharmProjects/Personal_Projects/Dynamic Pricing/artifacts/")


def preprocess_data(input_path, output_path):
    # Load data
    data = pd.read_csv(input_path)

    # Separate features and target
    X = data.drop(columns=['Historical_Cost_of_Ride'])
    y = data['Historical_Cost_of_Ride']


    numeric_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings',
                        'Expected_Ride_Duration']
    categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Transform and save the pipeline
    X_preprocessed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, f'{ARTIFACTS_DIR}preprocessor.pkl')

    # Save preprocessed data
    pd.DataFrame(X_preprocessed).to_csv(output_path, index=False)
    y.to_csv(f'{ARTIFACTS_DIR}target.csv', index=False)
