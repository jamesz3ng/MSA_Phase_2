import json, os, joblib
import numpy as np

def init():
    # Loads the model
    global model
    model_name = "model.pkl"
    # AZURE_MODEL_DIR is an Azure environment variable where scripts are stored in the cloud and should not be changed
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name)
    model = joblib.load(model_path)

def run(request):
    # Loads the input data, runs the model on it, and returns its predictions
    data = json.loads(request)
    data = np.array(data["data"])
    result = model.predict(data)
    # Return the regression prediction results directly
    return result.tolist()
