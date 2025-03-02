from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_xgboost.pkl")

# Load category mappings
try:
    with open("category_mappings.json", "r") as f:
        category_mappings = json.load(f)
except FileNotFoundError:
    category_mappings = {"Brand": {}, "Model": {}, "Fuel_Type": {}, "Transmission": {}}
    print("⚠️ Warning: category_mappings.json not found. Using empty mappings.")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", mappings=category_mappings)  # ✅ Pass mappings!

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form
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

        # Make prediction
        prediction = model.predict(df)

        return render_template("index.html", prediction=round(prediction[0], 2), mappings=category_mappings)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

