# EcoPackAI — Sustainable Packaging Recommendation System

EcoPackAI is an AI-powered sustainability intelligence platform designed to recommend eco-friendly packaging materials using advanced feature engineering, machine learning models, and database-backed analytics.

The system evaluates packaging materials based on environmental impact, durability, recyclability, and cost efficiency to support data-driven sustainability decisions.

---

# Problem Statement

The packaging industry significantly contributes to global carbon emissions and environmental waste. Businesses often lack data-driven tools to evaluate sustainable alternatives.

EcoPackAI solves this problem by:

- Analyzing sustainability metrics
- Predicting CO₂ emissions
- Evaluating cost efficiency
- Ranking materials based on environmental suitability

This enables smarter, greener packaging decisions.

---

# System Architecture Overview

EcoPackAI follows a structured ML pipeline:

1. Data Ingestion  
2. Data Cleaning & Feature Engineering  
3. Model Training & Optimization  
4. Evaluation & Visualization  
5. PostgreSQL Data Storage  
6. Analytical Query Layer  

The architecture supports scalability and backend integration.

---

# Data Intelligence & Processing

## Dataset Overview
- Total Records: 5,000 materials
- Total Features: 19
- Duplicates Removed: 0

---

## Engineered Sustainability Metrics

To enhance predictive power, 6 new domain-specific features were created:

- Strength-to-Weight Ratio
- Environmental Impact Score
- CO₂ Impact Index
- Cost Efficiency Index
- Durability Score
- Enhanced Material Suitability Score

These features allow the system to move beyond raw material attributes and quantify sustainability performance.

---

# Machine Learning Models

EcoPackAI uses ensemble learning techniques to achieve high predictive accuracy.

## CO₂ Emission Prediction
- Model: XGBoost
- R² Score: 0.9954
- RMSE: 0.2107
- MAE: 0.1463
- Training Time: 0.74s

## Cost Efficiency Prediction
- Model: Random Forest
- R² Score: 0.9989
- RMSE: 0.3815
- MAE: 0.1582
- Training Time: 1.32s

Both models achieve greater than 99% accuracy, making them suitable for production environments.

---

# Sustainability Rankings

Based on composite scoring, the top sustainable materials identified include:

1. Mushroom Mycelium  
2. Seaweed-Based Packaging  
3. Bamboo Fiber  
4. Recycled Paper  
5. Bioplastic (PLA)  
6. Hemp Fiber  
7. Cornstarch Packaging  
8. Recycled Cardboard  
9. Bagasse (Sugarcane)  
10. Molded Pulp  

These materials consistently show low CO₂ emission scores and high biodegradability.

---

# Installation & Execution

## 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib psycopg2-binary python-dotenv matplotlib seaborn
2. Run Full Pipeline
Bash
# Step 1 — Process raw data and engineer features
python process_data.py

# Step 2 — Train ML models
python train_models.py

# Step 3 — Load processed data into PostgreSQL
python load_to_db.py

# Step 4 — Generate evaluation reports and visualizations
python evaluate_models.py
PostgreSQL Integration
EcoPackAI integrates with PostgreSQL for structured storage and scalable querying.
```
## 2. Create Database
SQL
```bash
CREATE DATABASE ecopack_db;
Configure Environment Variables
Create a .env file:
```
###Code snippet
```bash
DB_NAME=ecopack_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```
##3. Load Data
```Bash
python load_to_db.py
The script:
```
##4. Connects to PostgreSQL

##5. Executes schema.sql

##6. Loads all processed material records

##7. Prepares database for analytical queries

---

#Example SQL Queries
##Top Sustainable Materials
###SQL
```bash
SELECT material_type, material_suitability_score
FROM materials
ORDER BY material_suitability_score DESC
LIMIT 10;
Low CO₂ Materials
```
###SQL
```bash SELECT material_type, co2_emission_score
FROM materials
WHERE co2_emission_score < 5
ORDER BY co2_emission_score;
```
#Project Structure
```bash
EcoPackAI/
├── process_data.py
├── train_models.py
├── load_to_db.py
├── evaluate_models.py
├── schema.sql
├── models/
├── reports/
└── requirements.txt
```
#Reporting & Visualization
Generated outputs include:

#Model performance comparison charts

### Feature importance graphs

### CO₂ prediction error analysis

### Sustainability ranking visualizations

### All reports are stored inside the reports/ directory.

#Production Readiness
EcoPackAI is structured for production deployment:

### High-accuracy ML models

### Modular data pipeline

### Database-backed architecture

### Environment-based configuration

### Reproducible training workflow

### The system can be extended into:

### REST API using Flask or FastAPI

### Sustainability analytics dashboard

### Carbon footprint simulation tool

### Enterprise procurement decision engine

#Future Enhancements
Real-time material recommendation API

Carbon emission forecasting

Supplier integration

Dashboard with business intelligence tools

# Cloud-based deployment

Cloud deployment using render and render postgres database instance

#License
MIT License
