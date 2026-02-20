import pandas as pd
import numpy as np

print("=== MODULE 2: DATA CLEANING & FEATURE ENGINEERING ===")

# -------------------------------------------------
# 1. LOAD DATA (NEW CLEAN FILES)
# -------------------------------------------------
products = pd.read_csv("data/products_ml_clean.csv")
materials = pd.read_csv("data/materials_ml_clean.csv")

print("✔ Products loaded:", products.shape)
print("✔ Materials loaded:", materials.shape)

# -------------------------------------------------
# 2. BASIC VALIDATION
# -------------------------------------------------
assert "material_type" in products.columns, "products file missing material_type"
assert "material_type" in materials.columns, "materials file missing material_type"

# -------------------------------------------------
# 3. MERGE PRODUCTS WITH MATERIAL PROPERTIES
# -------------------------------------------------
# This merge encodes historical association (PURE ML)
merged = products.merge(
    materials,
    on="material_type",
    how="left"
)

print("✔ Merged dataset shape:", merged.shape)

# -------------------------------------------------
# 4. HANDLE MISSING VALUES (ROBUST)
# -------------------------------------------------
for col in merged.columns:
    if merged[col].dtype == "object":
        merged[col] = merged[col].fillna(merged[col].mode()[0])
    else:
        merged[col] = merged[col].fillna(merged[col].median())

# -------------------------------------------------
# 5. FEATURE ENGINEERING (NO RULES)
# -------------------------------------------------

# Normalized load ratio (how heavy the product is relative to material capacity)
merged["load_ratio"] = (
    merged["product_weight_g"] / 1000
) / merged["weight_capacity_kg"]

# Environmental pressure score (learned later by ML)
merged["environmental_pressure"] = (
    merged["moisture_sensitivity"] * merged["moisture_barrier"] +
    merged["temperature_sensitivity"] * merged["temp_resistance"]
)

# Fragility–rigidity interaction (numeric, not rule-based)
merged["protection_score"] = (
    merged["fragility_level"] * merged["rigidity"]
)

# Sustainability signal (ML learns tradeoff)
merged["sustainability_index"] = (
    merged["biodegradability_score"] * 0.6 +
    merged["recyclability_pct"] * 0.4
)

# Shelf-life stress factor
merged["shelf_life_stress"] = (
    merged["shelf_life_days"] / 365
)

# -------------------------------------------------
# 6. FINAL TARGET VARIABLES (DATA-DRIVEN)
# -------------------------------------------------
# These are REGRESSION targets used later by ML
# (not hard constraints)

merged["cost_inr_per_kg"] = (
    merged["cost_inr_per_kg"] +
    merged["load_ratio"] * 8 +
    merged["rigidity"] * 5 +
    np.random.normal(0, 2, len(merged))
).clip(lower=1)

merged["co2_emission_per_kg"] = (
    merged["co2_emission_per_kg"] +
    merged["rigidity"] * 0.4 +
    (1 - merged["biodegradability_score"] / 10) * 0.6 +
    np.random.normal(0, 0.1, len(merged))
).clip(lower=0.01)

# -------------------------------------------------
# 7. DROP IDENTIFIERS (OPTIONAL, CLEAN ML INPUT)
# -------------------------------------------------
merged.drop(columns=["product_id"], inplace=True)

# -------------------------------------------------
# 8. SAVE FINAL DATASET
# -------------------------------------------------
output_path = "module2_final_cleaned_feature_engineered_dataset.csv"
merged.to_csv(output_path, index=False)

print("✔ Module 2 completed successfully")
print("✔ Output saved:", output_path)
