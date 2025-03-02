import json
import pandas as pd

# Load the dataset
file_path = "car_price_dataset.csv"
df = pd.read_csv(file_path)

# Extract categorical mappings
category_mappings = {}
for col in ['Brand', 'Model', 'Fuel_Type', 'Transmission']:
    unique_values = df[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    category_mappings[col] = mapping

# Save mappings as JSON
with open("category_mappings.json", "w") as f:
    json.dump(category_mappings, f, indent=4)

print("✅ Category mappings saved as category_mappings.json")
