
![example workflow](https://github.com/ash-sha/Dynamic_Pricing/actions/workflows/dynamic_pricing_pipeline.yml/badge.svg)

# Dynamic Pricing Model

Dynamic Pricing is an essential aspect of modern business operations, where prices are dynamically adjusted based on various factors such as demand, seasonality, competition, and customer behavior. This project leverages machine learning to implement a robust Dynamic Pricing model, enabling businesses to optimize their pricing strategies effectively.

---

## **Project Overview**

This project trains a machine learning model using a Gradient Boosting Regressor to predict optimal prices based on historical data. The model is managed and tracked with [MLflow](https://mlflow.org/), which facilitates versioning, artifact storage, and model serving.

---

## **Features**

- **Data Preprocessing**: Handles input data preparation, including feature extraction and splitting into training and testing sets.
- **Model Training**: Utilizes a Gradient Boosting Regressor for accurate price prediction.
- **Model Management**: Tracks experiments, logs metrics, and registers models using MLflow.
- **Dynamic Model Loading**: Automatically fetches and loads the latest production model for predictions.

---

## **Requirements**

To run this project, ensure you have the following installed:

- Python 3.8+
- Required Python packages (listed in `requirements.txt`)
- MLflow (for model tracking and management)
- Scikit-learn (for machine learning)
- Pandas (for data manipulation)

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Setup and Usage**

### **1. Data Preparation**
Prepare two CSV files:
- `batch_test_data.csv`: test data for batch inference.
- `single_test_data.csv`: test data for single inference.
- 
### **2. Train the Model**
Train and log the model using MLflow:


This script:
1. Loads the data.
2. Splits it into training and testing sets.
3. Trains a Gradient Boosting Regressor.
4. Logs the model and metrics (e.g., MAE) in MLflow.

### **3. Accessing the Model**
#### **Load from Run URI**
Retrieve and use the model from its run:
```python

model = mlflow.pyfunc.load_model((f"runs:/{run_id}/model")) #return run_id from train.py
predictions = model.predict(new_data)
```

#### **Load from Model Registry**
If registered in the Model Registry:
```python
model_uri = 'models:/dynamic_pricing_model/Production'
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(new_data)
```

---

## **Folder Structure**
```
Dynamic_Pricing/
|-- artifacts/
|-- scripts/
    |-- __init__.py            
    |-- train.py            # Script to train and log the model
    |-- inference.py            # Script for single and batch inference
    |-- main.py            # Script to run the pipeline
    |-- preprocess.py            # Script to preprocess
|-- requirements.txt          # Dependencies for the project
|-- mlruns/                   # MLflow tracking directory
|-- data/                     # Sample input data
|-- README.md                 # Project documentation
```

---

## **MLflow Tracking and Management**

To visualize and manage MLflow runs:
1. Start the MLflow UI:
   ```bash
   mlflow ui
   ```
2. Navigate to [http://localhost:5000](http://localhost:5000).
3. View logged runs, metrics, and artifacts.

---

## **Acknowledgements**

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## **License**

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## **Future Enhancements**

- Support for additional machine learning models.
- Integration with real-time pricing APIs.
- Enhanced data preprocessing pipelines.
- Deployment of the model for real-time inference.

