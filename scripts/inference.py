import os

import joblib
import mlflow.pyfunc
import pandas as pd

ARTIFACTS_DIR = os.getenv("PROCESSED_DIR","/Users/aswathshakthi/PycharmProjects/Personal_Projects/Dynamic Pricing/artifacts/")

def run_inference(input_path, output_path,run_id):

    model = mlflow.pyfunc.load_model((f"runs:/{run_id}/model"))

    preprocessor = joblib.load(f'{ARTIFACTS_DIR}preprocessor.pkl')
    data = pd.read_csv(input_path)
    data_transformed = preprocessor.transform(data)

    predictions = model.predict(data_transformed)
    pd.DataFrame(predictions, columns=["Predicted_Fare"]).to_csv(output_path, index=False)


def batch_inference(data_path, output_path,run_id):
    run_inference(data_path, output_path,run_id)
