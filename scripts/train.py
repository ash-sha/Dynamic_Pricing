import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model(data_path, target_path):
    # Load preprocessed data
    X = pd.read_csv(data_path)
    y = pd.read_csv(target_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    # Determine whether we're running in GitHub Actions or locally
    is_github_actions = os.getenv("GITHUB_ACTIONS") is not None

    # Set the artifact path based on the environment
    if is_github_actions:
        artifact_path = os.path.join(os.getenv("GITHUB_WORKSPACE"), "scripts/mlruns/model")
    else:
        artifact_path = "model"  # Local path or custom directory


    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Log model and metrics with MLflow
    with mlflow.start_run():
        mlflow.log_metric("MAE", mae)
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, registered_model_name="dynamic_pricing_model")
        print(f"Model logged at: {artifact_path}")
        run_id = mlflow.active_run().info.run_id

    return model, run_id
