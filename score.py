import json, os, joblib
import numpy as np

def init():
    # Logs are helpful for diagnostics. Azure ML captures logs from the print function.
    print("Initializing model...")
    global model
    model_name = "model.pkl"
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name)
    model = joblib.load(model_path)
    print("Model initialized!")

def run(request):
    try:
        # Parsing the request data
        data = json.loads(request)
        
        if "data" not in data:
            return {"error": "Invalid input. 'data' key is missing."}

        input_data = np.array(data["data"])

        # Making predictions
        result = model.predict(input_data)
        
        # Return the regression prediction results
        return result.tolist()

    except Exception as e:
        # Handle any type of error and return a descriptive message.
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"error": error_message}
