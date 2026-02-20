import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

print("="*80)
print("EcoPackAI - Comprehensive Evaluation Report Generator")
print("="*80)

# Load data
df = pd.read_csv('materials_processed_milestone1.csv')
print(f"\n✓ Loaded {len(df)} records")

# Load models
co2_model = joblib.load('models/co2_model.pkl')
cost_model = joblib.load('models/cost_model.pkl')
le_material = joblib.load('models/le_material.pkl')
print("✓ Loaded trained models")

# Prepare features
df['Material_Encoded'] = le_material.transform(df['Material_Type'])
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

# Make predictions
co2_predictions = co2_model.predict(X)
cost_predictions = cost_model.predict(X)

# Calculate metrics
co2_r2 = r2_score(df['CO2_Emission_Score'], co2_predictions)
co2_rmse = np.sqrt(mean_squared_error(df['CO2_Emission_Score'], co2_predictions))
cost_r2 = r2_score(df['Cost_Efficiency_Index'], cost_predictions)
cost_rmse = np.sqrt(mean_squared_error(df['Cost_Efficiency_Index'], cost_predictions))

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print(f"\nCO2 Emission Prediction:")
print(f"  R² Score: {co2_r2:.4f}")
print(f"  RMSE:     {co2_rmse:.4f}")
print(f"\nCost Efficiency Prediction:")
print(f"  R² Score: {cost_r2:.4f}")
print(f"  RMSE:     {cost_rmse:.4f}")

# Generate visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

if not os.path.exists('reports'):
    os.makedirs('reports')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Feature Importance (if available)
if hasattr(co2_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': co2_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance for CO2 Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved feature_importance.png")
    plt.close()

# 2. Top Materials Analysis
plt.figure(figsize=(12, 6))
top_10 = df.nlargest(10, 'Material_Suitability_Score')
sns.barplot(data=top_10, y='Material_Type', x='Material_Suitability_Score', palette='RdYlGn')
plt.title('Top 10 Materials by Suitability Score', fontsize=14, fontweight='bold')
plt.xlabel('Material Suitability Score')
plt.ylabel('Material Type')
plt.tight_layout()
plt.savefig('reports/top_materials.png', dpi=300, bbox_inches='tight')
print("✓ Saved top_materials.png")
plt.close()

# 3. Environmental Impact Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['CO2_Emission_Score'], bins=50, color='coral', edgecolor='black', alpha=0.7)
plt.title('CO2 Emission Score Distribution')
plt.xlabel('CO2 Emission Score')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['Biodegradability_Score'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Biodegradability Score Distribution')
plt.xlabel('Biodegradability Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('reports/environmental_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved environmental_distribution.png")
plt.close()

# 4. Material Type Analysis
material_stats = df.groupby('Material_Type').agg({
    'Material_Suitability_Score': 'mean',
    'CO2_Emission_Score': 'mean',
    'Biodegradability_Score': 'mean'
}).round(2).sort_values('Material_Suitability_Score', ascending=False).head(15)

plt.figure(figsize=(14, 8))
material_stats.plot(kind='barh', figsize=(14, 8))
plt.title('Average Scores by Material Type (Top 15)', fontsize=14, fontweight='bold')
plt.xlabel('Score')
plt.ylabel('Material Type')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('reports/material_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved material_comparison.png")
plt.close()

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"\nReports saved in 'reports/' directory")
print(f"  - feature_importance.png")
print(f"  - top_materials.png")
print(f"  - environmental_distribution.png")
print(f"  - material_comparison.png")
