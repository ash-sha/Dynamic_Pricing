import os
from preprocess import preprocess_data
from train import train_model
from inference import run_inference, batch_inference
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.getenv("DATA_DIR","/Users/aswathshakthi/PycharmProjects/Personal_Projects/Dynamic Pricing/data/")
ARTIFACTS_DIR = os.getenv("PROCESSED_DIR","/Users/aswathshakthi/PycharmProjects/Personal_Projects/Dynamic Pricing/artifacts/")

def main():
    # Step 1: Preprocess the Data
    input_data_path = f"{DATA_DIR}dynamic_pricing.csv"
    preprocessed_data_path = f"{ARTIFACTS_DIR}preprocessed_data.csv"
    target_data_path = f"{ARTIFACTS_DIR}target.csv"
    print("Preprocessing data...")
    preprocess_data(input_data_path, preprocessed_data_path)

    # Step 2: Train the Model
    print("Training model...")
    model,run_id = train_model(preprocessed_data_path, target_data_path)
    print("Model training complete.")

    # Step 3: Inference on a Single Record
    test_data_path = f"{DATA_DIR}/single_test_data.csv"  # Example file for testing inference
    inference_output_path = f"{ARTIFACTS_DIR}single_inference_output.csv"
    print("Running inference on a single record...")
    run_inference(test_data_path, inference_output_path,run_id)
    print(f"Inference results saved to {inference_output_path}")

    # Step 4: Batch Inference
    batch_test_data_path = f"{DATA_DIR}/batch_test_data.csv"  # Example batch test data
    batch_inference_output_path = f"{ARTIFACTS_DIR}batch_inference_output.csv"
    print("Running batch inference...")
    batch_inference(batch_test_data_path, batch_inference_output_path,run_id)
    print(f"Batch inference results saved to {batch_inference_output_path}")


if __name__ == "__main__":
    main()
