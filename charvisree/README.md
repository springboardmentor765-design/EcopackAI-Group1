# 🌱 EcoPackAI  
## AI-Powered Sustainable Packaging Recommendation System  

EcoPackAI is an AI-driven web application that recommends sustainable packaging materials based on product requirements, cost efficiency, and environmental impact.

The system uses machine learning models to predict packaging material cost and CO₂ emission impact, then ranks materials using a weighted suitability scoring mechanism.


## 🚀 Features

- Sustainable packaging recommendation  
- Cost prediction using Random Forest  
- CO₂ impact prediction using XGBoost  
- Weighted suitability scoring system  
- Interactive web interface  
- Data visualization and analytics  
- Cloud deployment using Replit  


## 🗂️ Datasets Used

### Material Dataset
Includes material properties such as:
- Strength (MPa)
- Weight Capacity (kg)
- Moisture Resistance
- Temperature Resistance
- Rigidity
- Biodegradability Score
- Recyclability Percentage
- CO₂ Emission per kg
- Cost per kg

### Product Dataset
Contains product attributes such as:
- Product Category
- Weight
- Volume
- Fragility Level
- Moisture Sensitivity
- Temperature Sensitivity
- Shelf Life
- Product Price


## 🧠 Machine Learning Models

- Random Forest Regressor – Cost Prediction  
- XGBoost Regressor – CO₂ Impact Prediction  

Evaluation Metrics:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  


## 🧠 Recommendation Logic

1. Filter feasible materials based on product constraints.  
2. Predict cost and CO₂ impact using ML models.  
3. Normalize prediction outputs.  
4. Compute Suitability Score:
   - CO₂ Impact – 40%
   - Cost Efficiency – 40%
   - Capacity Utilization – 20%
5. Rank materials based on suitability score.  
6. Display the best recommended packaging material.


## 🖥️ Web Interface

- Enter product details (category, weight, fragility, etc.)
- Click Predict to generate recommendations
- View ranked materials in tabular format
- Analyze cost and CO₂ comparison through charts


## ⚙️ Tech Stack

Backend: Python, Flask  
Frontend: HTML, CSS, Bootstrap  
Machine Learning: Scikit-learn, XGBoost  
Data Processing: Pandas, NumPy  
Database: PostgreSQL  
Deployment: Replit  


## 🚀 Live Demo

Replit Deployment Link:  
(Add your Replit live URL here)


## 📁 How to Run the Project Locally

1. Clone the repository  
2. Install dependencies using:  
   pip install -r requirements.txt  
3. Run the Flask application:  
   python app.py  
4. Open the local server link in your browser  


## 🌍 Objective

To promote sustainable packaging practices and help businesses make cost-effective and environmentally responsible decisions.


## 🔮 Future Enhancements

- Real-time sustainability analytics  
- Advanced carbon footprint modeling  
- Industry-specific packaging optimization  
- Integration with supply chain systems  


## 👨‍💻 Developed Under

Infosys Springboard Virtual Internship 6.0
