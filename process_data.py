import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EcoPackAI - Data Processing")
print("="*80)

# Load dataset
file_path = 'Ecopack-dataset_Ecopack-dataset.csv'
print(f"\nLoading dataset from {file_path}...")
df = pd.read_csv(file_path)
print(f"Loaded {len(df)} records with {len(df.columns)} columns")

print("\nDataset Info:")
print(df.info())
print("\nFirst 3 rows:")
print(df.head(3))

# Data quality check
print("\n" + "="*80)
print("Data Quality Check")
print("="*80)

missing_data = df.isnull().sum()
print("\nMissing Values:")
print(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values")

# Remove duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicates")

print("\nStatistical Summary:")
print(df.describe())

# Data cleaning
print("\n" + "="*80)
print("Data Cleaning")
print("="*80)

# Fill missing values with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {col} with median: {median_val:.2f}")

# Check for outliers
print("\nOutlier Detection (IQR Method):")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"  {col}: {outliers} outliers (kept for diversity)")

# Feature engineering
print("\n" + "="*80)
print("Feature Engineering")
print("="*80)

scaler = MinMaxScaler()

# Strength-to-weight ratio
print("\nCreating Strength-to-Weight Ratio...")
df['Strength_to_Weight_Ratio'] = df['Tensile_Strength_MPa'] / (df['Weight_Capacity_kg'] + 0.1)
df['Strength_to_Weight_Ratio_Normalized'] = scaler.fit_transform(df[['Strength_to_Weight_Ratio']]) * 100

# Environmental impact score (lower CO2 + higher biodegradability = better)
print("Creating Environmental Impact Score...")
df['Environmental_Impact_Raw'] = (
    (100 - df['CO2_Emission_Score']) * 0.5 +
    df['Biodegradability_Score'] * 0.3 +
    df['Recyclability_Percent'] * 0.2
)
df['Environmental_Impact_Score'] = scaler.fit_transform(df[['Environmental_Impact_Raw']]) * 100

# CO2 impact index (inverted - higher is better)
print("Creating CO2 Impact Index...")
df['CO2_Impact_Index_Raw'] = 1 / (df['CO2_Emission_Score'] + 0.1)
df['CO2_Impact_Index'] = scaler.fit_transform(df[['CO2_Impact_Index_Raw']]) * 100

# Cost efficiency
print("Creating Cost Efficiency Index...")
df['Cost_Efficiency_Raw'] = (
    (df['Recyclability_Percent'] + df['Biodegradability_Score']) / 
    (df['Weight_Capacity_kg'] + 0.01)
)
df['Cost_Efficiency_Index'] = scaler.fit_transform(df[['Cost_Efficiency_Raw']]) * 100

# Durability score
print("Creating Durability Score...")
df['Durability_Score'] = (
    df['Tensile_Strength_MPa'] * 0.6 +
    df['Moisture_Barrier_Grade'] * 10 * 0.4
)
df['Durability_Score_Normalized'] = scaler.fit_transform(df[['Durability_Score']]) * 100

# Final suitability score
print("Creating Material Suitability Score...")
df['Material_Suitability_Score'] = (
    0.30 * df['Environmental_Impact_Score'] +
    0.25 * df['CO2_Impact_Index'] +
    0.20 * df['Cost_Efficiency_Index'] +
    0.15 * df['Durability_Score_Normalized'] +
    0.10 * df['Strength_to_Weight_Ratio_Normalized']
)

# Validate features
print("\n" + "="*80)
print("Feature Validation")
print("="*80)

new_features = [
    'Strength_to_Weight_Ratio_Normalized',
    'Environmental_Impact_Score',
    'CO2_Impact_Index',
    'Cost_Efficiency_Index',
    'Durability_Score_Normalized',
    'Material_Suitability_Score'
]

print("\nNew Features Summary:")
print(df[new_features].describe())

# Check for invalid values
for feature in new_features:
    nan_count = df[feature].isna().sum()
    inf_count = np.isinf(df[feature]).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Warning: {feature} has {nan_count} NaN and {inf_count} infinite values")
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan).fillna(0)

# Material analysis
print("\n" + "="*80)
print("Material Analysis")
print("="*80)

print("\nTop 10 Materials by Suitability Score:")
top_materials = df.nlargest(10, 'Material_Suitability_Score')[
    ['Material_Type', 'Material_Suitability_Score', 'Environmental_Impact_Score', 
     'CO2_Emission_Score', 'Biodegradability_Score']
]
print(top_materials.to_string(index=False))

print("\nMaterial Type Distribution:")
material_counts = df['Material_Type'].value_counts()
print(material_counts.head(10))

# Save output
print("\n" + "="*80)
print("Saving Processed Data")
print("="*80)

output_path = 'materials_processed_milestone1.csv'
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
print(f"Total records: {len(df)}")
print(f"Total features: {len(df.columns)}")

# Save report
report_path = 'data_processing_report.txt'
with open(report_path, 'w') as f:
    f.write("EcoPackAI - Data Processing Report\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"Total Features: {len(df.columns)}\n")
    f.write(f"Duplicates Removed: {duplicates}\n\n")
    f.write("New Features:\n")
    f.write("- Strength-to-Weight Ratio\n")
    f.write("- Environmental Impact Score\n")
    f.write("- CO2 Impact Index\n")
    f.write("- Cost Efficiency Index\n")
    f.write("- Durability Score\n")
    f.write("- Material Suitability Score\n\n")
    f.write("Top 10 Materials:\n")
    f.write(top_materials.to_string(index=False))

print(f"Report saved to {report_path}")

print("\n" + "="*80)
print("Processing Complete")
print("="*80)
