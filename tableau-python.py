from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import joblib
# from tabpy.tabpy_tools.client import Client

# Load the trained models and encoders
clf_illicit = joblib.load("./model/clf_illicit.pkl")
reg_revenue = joblib.load("./model/reg_revenue.pkl")
label_encoders = joblib.load("./model/label_encoders.pkl")

# Define prediction functions
def predict_illicit(input_data):
    """
    Predict 'illicit' values using the trained classifier.
    """
    import numpy as np

    # Convert input data to a numpy array
    input_data = np.array(input_data, dtype=object)

    # Decode categorical features
    country_idx = 0  # country' is the first column
    office_id_idx = 1  # 'office.id' is the second column
    input_data[:, country_idx] = label_encoders['country'].transform(input_data[:, country_idx])
    input_data[:, office_id_idx] = label_encoders['office.id'].transform(input_data[:, office_id_idx])

    # Make predictions
    predictions = clf_illicit.predict(input_data)
    return {"data": predictions.tolist()}

def predict_revenue(input_data):
    """
    Predict 'revenue' values using the trained regressor.
    """
    print(f"Received input_data: {input_data}") # test code
    # Preprocess the input data
    input_data = np.array(input_data)
    
    # Decode categorical features
    country_idx = 0  # Assuming 'country' is the first column
    office_id_idx = 1  # Assuming 'office.id' is the second column
    input_data[:, country_idx] = label_encoders['country'].transform(input_data[:, country_idx])
    input_data[:, office_id_idx] = label_encoders['office.id'].transform(input_data[:, office_id_idx])

    predictions = reg_revenue.predict(input_data)
    return predictions.tolist()

# Deploy the function to TabPy
# if __name__ == "__main__":
#     client = Client("http://localhost:9004/")
#     client.deploy("PredictIllicit", predict_illicit, "Predicts illicit transactions (0 or 1).", override=True)