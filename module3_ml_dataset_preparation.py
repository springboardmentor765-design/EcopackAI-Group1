# ==========================================================
# EcoPackAI – Module 3
# Machine Learning Dataset Preparation
# ==========================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

print("=== MODULE 3: ML DATASET PREPARATION ===")

# ==========================================================
# STEP 1: Load Final Dataset from Module-2
# ==========================================================

df = pd.read_csv("module2_final_cleaned_feature_engineered_dataset.csv")

print("✔ Dataset Loaded Successfully")
print("✔ Shape:", df.shape)

# ==========================================================
# STEP 2: Select ML Features (INPUT VARIABLES)
# ==========================================================
# NOTE:
# - All features are numeric
# - No rules, no categories, no leakage
# - ML learns interactions from data

feature_cols = [
    # Product-related
    "product_weight_g",
    "product_volume_cm3",
    "fragility_level",
    "moisture_sensitivity",
    "temperature_sensitivity",
    "shelf_life_days",
    "price_inr",

    # Material intrinsic properties
    "strength_mpa",
    "weight_capacity_kg",
    "moisture_barrier",
    "temp_resistance",
    "rigidity",
    "biodegradability_score",
    "recyclability_pct",

    # Engineered interaction features (from Module 2)
    "load_ratio",
    "environmental_pressure",
    "protection_score",
    "sustainability_index",
    "shelf_life_stress"
]

X = df[feature_cols]

print("\n✔ Selected ML Features:")
for col in feature_cols:
    print("  -", col)

# ==========================================================
# STEP 3: Define Target Variables
# ==========================================================

# 🎯 Target 1: Packaging Cost Prediction
y_cost = df["cost_inr_per_kg"]

# 🎯 Target 2: CO₂ Emission Prediction
y_co2 = df["co2_emission_per_kg"]

print("\n✔ Targets Defined:")
print("  - cost_inr_per_kg")
print("  - co2_emission_per_kg")

# ==========================================================
# STEP 4: Train–Test Split
# ==========================================================

X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X,
    y_cost,
    y_co2,
    test_size=0.2,
    random_state=42
)

print("\n✔ Train/Test Split Completed")
print("  X_train:", X_train.shape)
print("  X_test :", X_test.shape)

# ==========================================================
# STEP 5: Scaling Pipeline (VERY IMPORTANT)
# ==========================================================
# Scaling is required because:
# - Weight, cost, CO₂, rigidity all have different magnitudes
# - ML models converge better with standardized data

scaler_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

X_train_scaled = scaler_pipeline.fit_transform(X_train)
X_test_scaled = scaler_pipeline.transform(X_test)

print("\n✔ Feature Scaling Applied (StandardScaler)")

# ==========================================================
# STEP 6: Convert Back to DataFrame
# ==========================================================

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

# ==========================================================
# STEP 7: Save ML-Ready Datasets
# ==========================================================

X_train_scaled_df.to_csv("X_train_ml.csv", index=False)
X_test_scaled_df.to_csv("X_test_ml.csv", index=False)

y_cost_train.to_csv("y_cost_train.csv", index=False)
y_cost_test.to_csv("y_cost_test.csv", index=False)

y_co2_train.to_csv("y_co2_train.csv", index=False)
y_co2_test.to_csv("y_co2_test.csv", index=False)

# Save scaler for Module 4 inference
joblib.dump(scaler_pipeline, "ml_feature_scaler.joblib")

print("\n✔ ML-ready datasets saved:")
print("  - X_train_ml.csv")
print("  - X_test_ml.csv")
print("  - y_cost_train.csv")
print("  - y_cost_test.csv")
print("  - y_co2_train.csv")
print("  - y_co2_test.csv")
print("  - ml_feature_scaler.joblib")

# ==========================================================
# STEP 8: Final Confirmation
# ==========================================================

print("\nModule 3 completed successfully 🚀")
