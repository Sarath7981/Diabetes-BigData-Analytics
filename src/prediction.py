
import joblib
import pandas as pd
import os

# Load the saved best model
model_path = "models/best_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Run model_training.py first.")

model = joblib.load(model_path)

# Input feature names and their expected order
feature_names = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]


def predict_diabetes(user_data):

    # Convert user input into a DataFrame
    input_df = pd.DataFrame([user_data], columns=feature_names)

    # Predict class
    prediction = model.predict(input_df)[0]

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        result = "Diabetes-positive class"
    else:
        result = "Diabetes-negative class"

    return {
        "prediction": int(prediction),
        "result": result,
        "probability": round(float(probability) * 100, 2)
    }


# Test prediction using sample input
if __name__ == "__main__":

    sample_input = [
        1,   # HighBP
        1,   # HighChol
        1,   # CholCheck
        30,  # BMI
        1,   # Smoker
        0,   # Stroke
        0,   # HeartDiseaseorAttack
        1,   # PhysActivity
        1,   # Fruits
        1,   # Veggies
        0,   # HvyAlcoholConsump
        1,   # AnyHealthcare
        0,   # NoDocbcCost
        3,   # GenHlth
        5,   # MentHlth
        5,   # PhysHlth
        0,    # DiffWalk
        1,   # Sex
        9,   # Age
        5,   # Education
        6    # Income
    ]

    result = predict_diabetes(sample_input)

    print("\n========== DIABETES PREDICTION =========")
    print("Prediction:", result["result"])
    print("Positive-class probability:", result["probability"], "%")
