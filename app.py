from flask import Flask, request
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
from sklearn.decomposition import PCA
import joblib

from sklearn.utils import resample
from flask_cors import CORS


# Step 4: Migraine Prevention Suggestions
def suggest_migraine_prevention(sleep_hours, stress_level, weather_change, dietary_habits, screen_time):
    suggestions = []
    if sleep_hours < 6:
        suggestions.append("Increase sleep hours to at least 7-8 hours per night.")
    if stress_level > 6:
        suggestions.append("Practice stress management techniques like meditation and deep breathing.")
    if weather_change == 1:
        suggestions.append("Stay hydrated and avoid sudden temperature changes.")
    if dietary_habits < 3:
        suggestions.append("Maintain a healthy diet, avoiding triggers like caffeine and processed foods.")
    if screen_time > 6:
        suggestions.append("Reduce screen time and take frequent breaks to rest your eyes.")

    return suggestions if suggestions else ["No major risk factors detected."]

# Step 5: User Interaction for Prediction
def get_user_input():
    print("Answer the following questions to assess your migraine risk:")
    sleep_hours = float(input("How many hours do you sleep per night? "))
    stress_level = int(input("On a scale of 1-10, how stressed do you feel daily? "))
    weather_change = int(input("Has there been a significant weather change recently? (1 for Yes, 0 for No): "))
    dietary_habits = int(input("How would you rate your diet? (1-Poor, 2-Average, 3-Good, 4-Excellent): "))
    screen_time = float(input("How many hours do you spend on screens daily? "))

    return np.array([[sleep_hours, stress_level, weather_change, dietary_habits, screen_time]])

# Step 6: Predict Migraine Risk
def predict_migraine(input_data_raw, best_model, scaler, pca, poly):
    input_data = poly.transform(input_data_raw)  # Apply Polynomial Features
    input_data = scaler.transform(input_data)
    input_data = pca.transform(input_data)

    predicted_severity = best_model.predict(input_data)[0]

    if predicted_severity > 5:
        status = "High risk of migraine."
    elif predicted_severity > 2:
        status = "Moderate risk of migraine."
    else:
        status = "Low risk of migraine."

    # Access the original features before transformation for suggestions
    original_features = input_data_raw[0]
    suggestions = suggest_migraine_prevention(*original_features)

    print("\nMigraine Prediction:", status)
    print("Suggestions to Reduce Risk:")

    return {"status": status, "suggestions": suggestions}

app = Flask(__name__)

CORS(app)

best_final_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
poly = joblib.load("poly.pkl")

@app.route("/predict", methods=["POST"])

def predict():
    data = request.get_json()
    print("data", data)
    sleep_hours = int(data.get("sleep"))
    stress_level = int(data.get("stress"))
    weather_change = int(data.get("weather"))
    dierary_habits = int(data.get("diet"))
    screen_time = int(data.get("screentime"))

    print(sleep_hours, stress_level, weather_change, dierary_habits, screen_time)


    input_data_raw = np.array([[sleep_hours, stress_level, weather_change, dierary_habits, screen_time]])

    # ✅ Make sure you call it only once
    return predict_migraine(input_data_raw, best_final_model, scaler, pca, poly)

if __name__ == "__main__":
    app.run()