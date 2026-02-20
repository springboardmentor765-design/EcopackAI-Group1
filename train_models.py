import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Check available libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

print("="*80)
print("EcoPackAI - Model Training")
print("="*80)
print(f"\nAvailable Libraries:")
print(f"  XGBoost: {'Yes' if HAS_XGB else 'No'}")
print(f"  LightGBM: {'Yes' if HAS_LGBM else 'No'}")
print(f"  CatBoost: {'Yes' if HAS_CATBOOST else 'No'}")

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Load processed data
input_path = 'materials_processed_milestone1.csv'
print(f"\nLoading data from {input_path}...")
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} records with {len(df.columns)} features")

# Prepare data
print("\n" + "="*80)
print("Data Preparation")
print("="*80)

# Encode material types
le_material = LabelEncoder()
df['Material_Encoded'] = le_material.fit_transform(df['Material_Type'])

# Select features
feature_cols = [
    'Material_Encoded',
    'Tensile_Strength_MPa',
    'Weight_Capacity_kg',
    'Biodegradability_Score',
    'Recyclability_Percent',
    'Moisture_Barrier_Grade',
    'Strength_to_Weight_Ratio_Normalized',
    'Environmental_Impact_Score',
    'Durability_Score_Normalized'
]

X = df[feature_cols]
print(f"\nFeatures: {len(feature_cols)}")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

# Target variables
y_co2 = df['CO2_Emission_Score']
y_cost_eff = df['Cost_Efficiency_Index']

# Split data (80% train, 20% test)
X_train, X_test, y_co2_train, y_co2_test, y_cost_train, y_cost_test = train_test_split(
    X, y_co2, y_cost_eff, test_size=0.2, random_state=42
)

print(f"\nData Split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing: {len(X_test)} samples")

# Model evaluation function
def evaluate_model(name, y_true, y_pred, training_time):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Time:     {training_time:.2f}s")
    print(f"{'='*60}")
    
    return {'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'time': training_time}

# Store results
co2_results = []
cost_results = []

# Train CO2 prediction models
print("\n" + "="*80)
print("CO2 EMISSION PREDICTION")
print("="*80)

# Random Forest
print("\n[1/4] Training Random Forest...")
start_time = time.time()
rf_co2 = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_co2.fit(X_train, y_co2_train)
y_co2_pred_rf = rf_co2.predict(X_test)
rf_time = time.time() - start_time
co2_results.append(evaluate_model("Random Forest (CO2)", y_co2_test, y_co2_pred_rf, rf_time))

# XGBoost
if HAS_XGB:
    print("\n[2/4] Training XGBoost...")
    start_time = time.time()
    xgb_co2 = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_co2.fit(X_train, y_co2_train)
    y_co2_pred_xgb = xgb_co2.predict(X_test)
    xgb_time = time.time() - start_time
    co2_results.append(evaluate_model("XGBoost (CO2)", y_co2_test, y_co2_pred_xgb, xgb_time))

# LightGBM
if HAS_LGBM:
    print("\n[3/4] Training LightGBM...")
    start_time = time.time()
    lgbm_co2 = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_co2.fit(X_train, y_co2_train)
    y_co2_pred_lgbm = lgbm_co2.predict(X_test)
    lgbm_time = time.time() - start_time
    co2_results.append(evaluate_model("LightGBM (CO2)", y_co2_test, y_co2_pred_lgbm, lgbm_time))

# CatBoost
if HAS_CATBOOST:
    print("\n[4/4] Training CatBoost...")
    start_time = time.time()
    cat_co2 = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=8,
        random_state=42,
        verbose=0
    )
    cat_co2.fit(X_train, y_co2_train)
    y_co2_pred_cat = cat_co2.predict(X_test)
    cat_time = time.time() - start_time
    co2_results.append(evaluate_model("CatBoost (CO2)", y_co2_test, y_co2_pred_cat, cat_time))

# Train cost efficiency models
print("\n" + "="*80)
print("COST EFFICIENCY PREDICTION")
print("="*80)

# Random Forest
print("\n[1/4] Training Random Forest...")
start_time = time.time()
rf_cost = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_cost.fit(X_train, y_cost_train)
y_cost_pred_rf = rf_cost.predict(X_test)
rf_time = time.time() - start_time
cost_results.append(evaluate_model("Random Forest (Cost)", y_cost_test, y_cost_pred_rf, rf_time))

# XGBoost
if HAS_XGB:
    print("\n[2/4] Training XGBoost...")
    start_time = time.time()
    xgb_cost = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_cost.fit(X_train, y_cost_train)
    y_cost_pred_xgb = xgb_cost.predict(X_test)
    xgb_time = time.time() - start_time
    cost_results.append(evaluate_model("XGBoost (Cost)", y_cost_test, y_cost_pred_xgb, xgb_time))

# LightGBM
if HAS_LGBM:
    print("\n[3/4] Training LightGBM...")
    start_time = time.time()
    lgbm_cost = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_cost.fit(X_train, y_cost_train)
    y_cost_pred_lgbm = lgbm_cost.predict(X_test)
    lgbm_time = time.time() - start_time
    cost_results.append(evaluate_model("LightGBM (Cost)", y_cost_test, y_cost_pred_lgbm, lgbm_time))

# CatBoost
if HAS_CATBOOST:
    print("\n[4/4] Training CatBoost...")
    start_time = time.time()
    cat_cost = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=8,
        random_state=42,
        verbose=0
    )
    cat_cost.fit(X_train, y_cost_train)
    y_cost_pred_cat = cat_cost.predict(X_test)
    cat_time = time.time() - start_time
    cost_results.append(evaluate_model("CatBoost (Cost)", y_cost_test, y_cost_pred_cat, cat_time))

# Find best models
print("\n" + "="*80)
print("Model Comparison")
print("="*80)

best_co2 = max(co2_results, key=lambda x: x['r2'])
best_cost = max(cost_results, key=lambda x: x['r2'])

print(f"\nBest CO2 Model: {best_co2['model']}")
print(f"  R² Score: {best_co2['r2']:.4f}")
print(f"  RMSE: {best_co2['rmse']:.4f}")

print(f"\nBest Cost Model: {best_cost['model']}")
print(f"  R² Score: {best_cost['r2']:.4f}")
print(f"  RMSE: {best_cost['rmse']:.4f}")

# Save models
print("\n" + "="*80)
print("Saving Models")
print("="*80)

# Save best available models
if HAS_XGB:
    joblib.dump(xgb_co2, 'models/co2_model.pkl')
    joblib.dump(xgb_cost, 'models/cost_model.pkl')
    print("Saved XGBoost models")
elif HAS_LGBM:
    joblib.dump(lgbm_co2, 'models/co2_model.pkl')
    joblib.dump(lgbm_cost, 'models/cost_model.pkl')
    print("Saved LightGBM models")
else:
    joblib.dump(rf_co2, 'models/co2_model.pkl')
    joblib.dump(rf_cost, 'models/cost_model.pkl')
    print("Saved Random Forest models")

joblib.dump(le_material, 'models/le_material.pkl')
print("Saved encoders")

# Save report
report_path = 'model_training_report.txt'
with open(report_path, 'w') as f:
    f.write("EcoPackAI - Model Training Report\n")
    f.write("="*80 + "\n\n")
    f.write("CO2 PREDICTION:\n")
    f.write("-"*80 + "\n")
    for result in co2_results:
        f.write(f"{result['model']:25} | R²: {result['r2']:.4f} | RMSE: {result['rmse']:.4f} | Time: {result['time']:.2f}s\n")
    
    f.write("\n\nCOST EFFICIENCY PREDICTION:\n")
    f.write("-"*80 + "\n")
    for result in cost_results:
        f.write(f"{result['model']:25} | R²: {result['r2']:.4f} | RMSE: {result['rmse']:.4f} | Time: {result['time']:.2f}s\n")
    
    f.write(f"\n\nBEST MODELS:\n")
    f.write("-"*80 + "\n")
    f.write(f"CO2: {best_co2['model']} (R² = {best_co2['r2']:.4f})\n")
    f.write(f"Cost: {best_cost['model']} (R² = {best_cost['r2']:.4f})\n")

print(f"Report saved to {report_path}")

print("\n" + "="*80)
print("Training Complete")
print("="*80)
