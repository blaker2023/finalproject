from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import pickle  # Using pickle for model loading
import os
import xgboost as xgb  # Ensure XGBoost is imported

# Initialize Flask app
app = Flask(__name__)

# ✅ Define model file paths
MODEL_PATH_PKL = "best_xgboost_model.pkl"
MODEL_PATH_JSON = "best_xgboost_model.json"

# ✅ Load the trained model (Handle both pickle & XGBoost Booster formats)
model = None
try:
    if os.path.exists(MODEL_PATH_PKL):
        print("✅ Loading model from pickle .pkl file...")
        with open(MODEL_PATH_PKL, "rb") as f:
            model = pickle.load(f)  # Load scikit-learn compatible model
    elif os.path.exists(MODEL_PATH_JSON):
        print("✅ Loading model from XGBoost JSON file...")
        model = xgb.Booster()
        model.load_model(MODEL_PATH_JSON)  # Load Booster model
    else:
        raise FileNotFoundError("⚠️ Model file not found! Please train and save the model.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ Load category mappings
CATEGORY_MAPPING_PATH = "category_mappings.json"
try:
    with open(CATEGORY_MAPPING_PATH, "r") as f:
        category_mappings = json.load(f)
    print("✅ Category mappings loaded successfully.")
except FileNotFoundError:
    category_mappings = {"Brand": {}, "Model": {}, "Fuel_Type": {}, "Transmission": {}}
    print("⚠️ Warning: category_mappings.json not found. Using empty mappings.")

@app.route("/")
def home():
    """Render the homepage."""
    return render_template("index.html", mappings=category_mappings)

@app.route("/predict", methods=["POST"])
def predict():
    """Handle car price predictions."""
    if model is None:
        return jsonify({"error": "Model file not found. Ensure 'best_xgboost_model.pkl' or 'best_xgboost_model.json' exists."})

    try:
        # Get input data from form
        data = {
            "Brand": int(request.form["Brand"]),
            "Model": int(request.form["Model"]),
            "Year": int(request.form["Year"]),
            "Engine_Size": float(request.form["Engine_Size"]),
            "Fuel_Type": int(request.form["Fuel_Type"]),
            "Transmission": int(request.form["Transmission"]),
            "Mileage": int(request.form["Mileage"]),
            "Doors": int(request.form["Doors"]),
            "Owner_Count": int(request.form["Owner_Count"])
        }

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # ✅ Ensure correct feature order before prediction
        if hasattr(model, "feature_names_in_"):  # If using XGBRegressor or XGBClassifier
            expected_features = model.feature_names_in_
            df = df[expected_features]  # Align input features with model
            prediction = model.predict(df)

        elif isinstance(model, xgb.Booster):  # If using XGBoost Booster
            prediction = model.predict(xgb.DMatrix(df))  # Convert to DMatrix for Booster model

        else:
            return jsonify({"error": "Model type not recognized. Ensure correct training and saving."})

        return render_template("index.html", prediction=round(float(prediction[0]), 2), mappings=category_mappings)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
