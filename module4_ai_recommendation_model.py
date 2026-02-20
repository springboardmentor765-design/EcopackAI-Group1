# module4_ai_recommendation_model.py
# ==========================================================
# EcoPackAI – Module 4 (FIXED: scaler feature-order mismatch)
# ==========================================================

import os
import sys
import pandas as pd
import numpy as np
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ===============================
# CONFIG
# ===============================
DATASET_PATH = "module2_final_cleaned_feature_engineered_dataset.csv"
MATERIALS_PATH = "data/materials_ml_clean.csv"
SCALER_PATH = "ml_feature_scaler.joblib"
XTRAIN_HEADER_PATH = "X_train_ml.csv"   # used only to read column order
MODEL_PATH = "material_recommender_model.joblib"
ENCODER_PATH = "category_encoder.joblib"



MAX_TRAIN_ROWS = 15000
NEGATIVE_SAMPLES = 2
TOP_K = 5
RANDOM_STATE = 42

ALLOWED_CATEGORIES = [
    "food", "electronics", "metal_part",
    "liquid", "cosmetics", "pharmaceutical"
]

CATEGORY_ALIASES = {
    "metal": "metal_part",
    "liquids": "liquid",
    "medicine": "pharmaceutical",
    "cosmetic": "cosmetics"
}

# ===============================
# HELPERS
# ===============================
def load_expected_numeric_feature_order():
    """
    Read header of X_train_ml.csv saved by Module 3 to obtain the exact
    numeric feature order that the scaler was fitted on.
    """
    if not os.path.exists(XTRAIN_HEADER_PATH):
        print(f"⚠ Warning: {XTRAIN_HEADER_PATH} not found. Falling back to default numeric feature order.")
        return None
    header = pd.read_csv(XTRAIN_HEADER_PATH, nrows=0)
    return list(header.columns)

# safe numeric conversion helper
def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

# ===============================
# LOAD DATA + SCALER
# ===============================
print("\n=== MODULE 4: PURE ML PACKAGING RECOMMENDER (FIXED) ===\n")

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset missing: {DATASET_PATH}")
    sys.exit(1)
if not os.path.exists(MATERIALS_PATH):
    print(f"ERROR: Materials file missing: {MATERIALS_PATH}")
    sys.exit(1)
if not os.path.exists(SCALER_PATH):
    print(f"ERROR: Scaler missing: {SCALER_PATH} (run Module 3 first)")
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)
materials = pd.read_csv(MATERIALS_PATH)
scaler = joblib.load(SCALER_PATH)

print(f"✔ Dataset loaded: {df.shape}")
print(f"✔ Materials loaded: {materials.shape}")

# sample training data for speed (same as before)
df = df.sample(n=min(MAX_TRAIN_ROWS, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

# ===============================
# DETERMINE SCALER FEATURE ORDER
# ===============================
expected_numeric_order = load_expected_numeric_feature_order()
if expected_numeric_order is None:
    # Fallback numeric list (should match Module 3 feature_cols order)
    expected_numeric_order = [
        "product_weight_g",
        "product_volume_cm3",
        "fragility_level",
        "moisture_sensitivity",
        "temperature_sensitivity",
        "shelf_life_days",
        "price_inr",
        "strength_mpa",
        "weight_capacity_kg",
        "moisture_barrier",
        "temp_resistance",
        "rigidity",
        "biodegradability_score",
        "recyclability_pct",
        "load_ratio",
        "environmental_pressure",
        "protection_score",
        "sustainability_index",
        "shelf_life_stress"
    ]
    print("⚠ Using fallback numeric feature order. Better to run Module 3 to create X_train_ml.csv.")

# Ensure all expected columns are present in pair_df later — we will check dynamically

# ===============================
# ENCODE CATEGORY (NOT SCALED)
# ===============================
if os.path.exists(ENCODER_PATH):
    le_category = joblib.load(ENCODER_PATH)
else:
    le_category = LabelEncoder()
    le_category.fit(df["product_category"])
    joblib.dump(le_category, ENCODER_PATH)

df["product_category_enc"] = le_category.transform(df["product_category"])


# ===============================
# BUILD PAIRWISE TRAINING DATA
# ===============================
pairs = []
material_names = materials["material_type"].tolist()

for _, row in df.iterrows():
    product_part = {
        "product_category_enc": row["product_category_enc"],
        "product_weight_g": row["product_weight_g"],
        "product_volume_cm3": row["product_volume_cm3"],
        "fragility_level": row["fragility_level"],
        "moisture_sensitivity": row["moisture_sensitivity"],
        "temperature_sensitivity": row["temperature_sensitivity"],
        "shelf_life_days": row["shelf_life_days"],
        "price_inr": row["price_inr"],
        "load_ratio": row.get("load_ratio", (row["product_weight_g"]/1000) / max(1.0, row.get("weight_capacity_kg", 1.0))),
        "environmental_pressure": row.get("environmental_pressure", 0.0),
        "protection_score": row.get("protection_score", 0.0),
        "sustainability_index": row.get("sustainability_index", 0.0),
        "shelf_life_stress": row.get("shelf_life_stress", row["shelf_life_days"]/365 if row["shelf_life_days"]>0 else 0.0)
    }

    mat = row["material_type"]
    if mat not in material_names:
        continue

    pos = product_part.copy()
    pos.update(materials[materials["material_type"] == mat].iloc[0].to_dict())
    pos["label"] = 1
    pairs.append(pos)

    neg_samples = random.sample([m for m in material_names if m != mat], min(NEGATIVE_SAMPLES, len(material_names)-1))
    for neg_mat in neg_samples:
        neg = product_part.copy()
        neg.update(materials[materials["material_type"] == neg_mat].iloc[0].to_dict())
        neg["label"] = 0
        pairs.append(neg)

pair_df = pd.DataFrame(pairs)
print(f"✔ Pairwise training data built: {pair_df.shape}")

# ===============================
# PREPARE TRAINING X (use exact numeric order)
# ===============================
# build numeric matrix using expected_numeric_order but guard if columns missing
missing = [c for c in expected_numeric_order if c not in pair_df.columns]
if missing:
    print("⚠ Warning: The following expected numeric columns are missing in pair_df:", missing)
    print("Attempting to fill missing columns with zeros to keep order consistent.")
    for c in missing:
        pair_df[c] = 0.0

# Extract numeric array in the expected order
X_num_df = pair_df[expected_numeric_order]
# Transform numeric features with scaler (the scaler expects this exact column order)
X_num_scaled = scaler.transform(X_num_df)

# categorical features (we keep only product_category_enc as before)
X_cat = pair_df[["product_category_enc"]].values

# final training matrix (category first, then scaled numeric)
X_train_matrix = np.hstack([X_cat, X_num_scaled])
y = pair_df["label"].values

# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=16,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

if os.path.exists(MODEL_PATH):
    print("✔ Loading existing trained model")
    model = joblib.load(MODEL_PATH)
else:
    print("Training ML model... ⏳")
    model.fit(X_train_matrix, y)
    joblib.dump(model, MODEL_PATH)
    print("✔ ML model trained and saved")


# ===============================
# USER INPUT (INFERENCE)
# ===============================
print("\n--- ENTER PRODUCT DETAILS ---")
print("Available categories:", ", ".join(ALLOWED_CATEGORIES))

raw_cat = input("Category: ").strip().lower()
category = CATEGORY_ALIASES.get(raw_cat, raw_cat)
if category not in ALLOWED_CATEGORIES:
    print("⚠ Unknown category, defaulting to 'food'")
    category = "food"
cat_enc = le_category.transform([category])[0]

def get_int(prompt, min_v=0, max_v=None):
    while True:
        try:
            val = int(input(prompt))
            if max_v is not None and val > max_v:
                print(f"Value must be ≤ {max_v}")
                continue
            return val
        except:
            print("Enter a valid integer")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except:
            print("Enter a valid number")

weight = get_float("Weight (g): ")
volume = get_float("Volume (cm3): ")
fragility = get_int("Fragility (0–2): ", 0, 2)
moisture = get_int("Moisture sensitivity (0–2): ", 0, 2)
temperature = get_int("Temperature sensitivity (0–2): ", 0, 2)
shelf_life = get_int("Shelf life (days): ")
price = get_float("Product price (INR): ")

# prepare candidate rows for each material in correct order
rows = []
for _, m in materials.iterrows():
    # numeric values must be in the same order as expected_numeric_order
    numeric_values_map = {
        "product_weight_g": weight,
        "product_volume_cm3": volume,
        "fragility_level": fragility,
        "moisture_sensitivity": moisture,
        "temperature_sensitivity": temperature,
        "shelf_life_days": shelf_life,
        "price_inr": price,
        "strength_mpa": m["strength_mpa"],
        "weight_capacity_kg": m["weight_capacity_kg"],
        "moisture_barrier": m["moisture_barrier"],
        "temp_resistance": m["temp_resistance"],
        "rigidity": m["rigidity"],
        "biodegradability_score": m["biodegradability_score"],
        "recyclability_pct": m["recyclability_pct"],
        # engineered features (recompute using material properties)
        "load_ratio": (weight / 1000) / m["weight_capacity_kg"] if m["weight_capacity_kg"] else 0.0,
        "environmental_pressure": moisture * m["moisture_barrier"] + temperature * m["temp_resistance"],
        "protection_score": fragility * m["rigidity"],
        "sustainability_index": m["biodegradability_score"] * 0.6 + m["recyclability_pct"] * 0.4,
        "shelf_life_stress": shelf_life / 365.0
    }

    # ensure we build numeric list in expected order
    numeric_list = [safe_float(numeric_values_map.get(c, 0.0)) for c in expected_numeric_order]

    # scale numeric_list with scaler (as 2D array)
    scaled_num = scaler.transform([numeric_list])[0]

    # construct final row: categorical first then scaled numeric
    final_row = np.hstack([[cat_enc], scaled_num])

    prob = model.predict_proba([final_row])[0][1]

    rows.append({
        "material_type": m["material_type"],
        "Suitability_Prob": prob,
        "cost_inr_per_kg": m.get("cost_inr_per_kg", np.nan),
        "co2_emission_per_kg": m.get("co2_emission_per_kg", np.nan),
        "recyclability_pct": m.get("recyclability_pct", np.nan),
        "biodegradability_score": m.get("biodegradability_score", np.nan)
    })

result = pd.DataFrame(rows).sort_values("Suitability_Prob", ascending=False).head(TOP_K)

# ===============================
# OUTPUT
# ===============================
print("\n=== AI RECOMMENDATION RESULT (PURE ML) ===")
print(f"✔ Recommended Material: {result.iloc[0]['material_type']}\n")
print(result.to_string(index=False))

print("\nModule 4 completed successfully 🚀")
