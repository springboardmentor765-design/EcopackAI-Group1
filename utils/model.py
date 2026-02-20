import joblib
import pandas as pd
import numpy as np
import os

MODEL_DIR = 'models'

class ModelManager:
    def __init__(self):
        self.co2_model = None
        self.cost_model = None
        self.encoder = None
        self._load_models()

    def _load_models(self):
        try:
            print("Loading models...")
            self.co2_model = joblib.load(os.path.join(MODEL_DIR, 'best_co2_model.pkl'))
            self.cost_model = joblib.load(os.path.join(MODEL_DIR, 'best_cost_model.pkl'))
            self.encoder = joblib.load(os.path.join(MODEL_DIR, 'le_material.pkl'))
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def get_available_materials(self):
        if self.encoder:
            return list(self.encoder.classes_)
        return []

    def predict_all(self, user_input):
        """
        Runs prediction for ALL available material types based on extended user inputs.
        user_input: dict containing weight_capacity, fragility, shelf_life, is_bulky, etc.
        """
        if not self.co2_model or not self.cost_model:
            return []

        materials = self.get_available_materials()
        raw_results = []

        # Extract user constraints with safe defaults
        target_weight = float(user_input.get('weight_capacity', 1.0))
        fragility_score = int(user_input.get('fragility_score', 0)) # 0-10
        shelf_life_days = int(user_input.get('shelf_life_days', 0))
        is_bulky = user_input.get('is_bulky', False)
        category = user_input.get('category', 'General')

        # Base Stats (simulated DB fetch) 
        # Corrected keys based on actual dataset classes
        material_stats = {
            'Cardboard Box': [45.0, 90.0, 95.0, 1],
            'Recycled Cardboard': [45.0, 90.0, 95.0, 1], # Added correct key
            'Bioplastic (PLA)': [35.0, 95.0, 80.0, 2],
            'Kraft Paper': [40.0, 98.0, 100.0, 1],
            'Mushroom Mycelium': [20.0, 100.0, 100.0, 1],
            'Corrugated Box': [60.0, 85.0, 90.0, 2],
            'Recycled Plastic (rPET)': [70.0, 10.0, 60.0, 3], 
            'Starch-based Loose Fill': [25.0, 100.0, 100.0, 1],
            'Glass Jar': [80.0, 0.0, 100.0, 5],
            'Aluminum Foil': [90.0, 0.0, 70.0, 5], # Reduced from 95 (User contamination issue)
            'Bagasse': [30.0, 100.0, 100.0, 1],
            'Virgin Plastic (PP)': [80.0, 0.0, 20.0, 5]
        }

        feature_names = [
            'Material_Encoded', 'Tensile_Strength_MPa', 'Weight_Capacity_kg', 
            'Biodegradability_Score', 'Recyclability_Percent', 'Moisture_Barrier_Grade'
        ]

        # 1. Collect Raw Predictions
        for mat in materials:
            try:
                encoded_mat = self.encoder.transform([mat])[0]
                stats = material_stats.get(mat, [50.0, 50.0, 50.0, 1])
                tensile, bio, recycl, moisture = stats

                X_input = pd.DataFrame([[encoded_mat, tensile, target_weight, bio, recycl, moisture]], 
                                       columns=feature_names)
                
                # Raw intrinsic predictions (Material properties)
                raw_co2 = self.co2_model.predict(X_input)[0]
                raw_cost = self.cost_model.predict(X_input)[0] # Efficiency Index

                # --- PHYSICS SIMULATION LAYER (The "Real World" Fix) ---
                # Calculate how MUCH material is needed based on Weight vs Strength.
                # Heuristic: We need a Safety Factor. 
                # If Tensile is low and Weight is high, we need THICKER packaging.
                # Factor = (Target Load) / (Material Strength * Constant)
                # We assume baseline is 1kg load for these stats.
                
                # Usage Factor: How many "units" of packaging relative to baseline?
                # Stronger materials (High Tensile) need LESS material.
                usage_factor = max(0.5, (target_weight * 5.0) / max(10, tensile))
                
                # Apply Factor to Predictions
                # CO2 Emissions multiply by usage (More material = More CO2)
                adj_co2 = raw_co2 * usage_factor
                
                # Cost Efficiency divides by usage (More material = Less Efficient)
                adj_cost = raw_cost / usage_factor  

                raw_results.append({
                    "material_type": mat,
                    "pred_co2": adj_co2,
                    "pred_cost": adj_cost,
                    "bio": bio, "recycl": recycl, "tensile": tensile, "moisture": moisture
                })
            except Exception as e:
                continue

        if not raw_results:
            return []

        # 2. Normalize and Rank
        co2_vals = [r['pred_co2'] for r in raw_results]
        cost_vals = [r['pred_cost'] for r in raw_results]
        
        min_co2, max_co2 = min(co2_vals), max(co2_vals)
        min_cost, max_cost = min(cost_vals), max(cost_vals)
        
        range_co2 = max_co2 - min_co2 if max_co2 != min_co2 else 1.0
        range_cost = max_cost - min_cost if max_cost != min_cost else 1.0

        final_ranking = []

        for r in raw_results:
            # Normalize (0 to 100 Scale)
            n_co2 = 100 * (1 - ((r['pred_co2'] - min_co2) / range_co2))
            n_cost = 100 * ((r['pred_cost'] - min_cost) / range_cost)

            # --- Expert System Logic ---
            penalty = 0
            
            # Shelf Life Logic (Moisture)
            # Penalize materials with poor moisture barrier for long shelf life
            if shelf_life_days > 7 and r['moisture'] < 3: 
                penalty += 60 # STRICTER Penalty (Rotting is fatal)
            
            # Sustainability Bias (Overkill Logic)
            # If light and short-lived, punish heavy/industrial materials
            if fragility_score < 4 and shelf_life_days < 10 and target_weight < 2.0:
                if r['material_type'] in ['Glass Jar', 'Aluminum Foil', 'Virgin Plastic (PP)']:
                    penalty += 40 
            
            # --- General Category Diversity Logic (Anti-Metal Bias) ---
            # If Category is General/Cosmetics, penalize Metal/Glass heavily
            if category in ['General', 'Cosmetics', 'Books']:
                 if r['material_type'] in ['Glass Jar', 'Aluminum Foil', 'Virgin Plastic (PP)']:
                    penalty += 50 # Increased from 25. Aluminum is not for books.

            # --- BONUS LOGIC (Pro-Eco Bias) ---
            # Reward Cardboard/Paper/Bio for appropriate scenarios (Short Shelf Life only)
            if shelf_life_days < 30:
                if r['material_type'] in ['Cardboard Box', 'Recycled Cardboard', 'Kraft Paper']:
                    if category in ['General', 'Books', 'Food']:
                        penalty -= 25 
                
                if r['material_type'] in ['Mushroom Mycelium', 'Bioplastic (PLA)', 'Bagasse']:
                    if category in ['Cosmetics', 'Food', 'General']:
                        penalty -= 20

            # Final Score Calculation
            
            # --- CAPPED UTILITY LOGIC (The "Smart Engineer" Fix) ---
            # Excess Strength is not valuable. If we only need 5 MPa, 90 MPa gives no extra bonus.
            # We define 'Useful Strength' relative to the Weight.
            # Base requirement: Weight * 3. Buffer: +15.
            required_tensile = (target_weight * 3.0) + 15
            useful_tensile = min(r['tensile'], required_tensile)
            
            # Scale useful tensile to 0-100 relative to max possible useful
            # (Assuming max useful ever needed is approx 90 for heavy items)
            phys_score = (useful_tensile / 90.0) * 100 
            
            sust_score = (r['bio'] + r['recycl'] + n_co2) / 3
            
            # Weighted: 40% Sustainability, 30% Cost, 30% Physics
            # We boost Sustainability weight to satisfy Eco-focus
            weighted_score = (0.45 * sust_score) + (0.25 * n_cost) + (0.30 * phys_score)
            
            final_score = max(0, min(100, weighted_score - penalty))

            final_ranking.append({
                "material_type": str(r['material_type']),
                "predicted_co2": float(round(r['pred_co2'], 2)),
                "predicted_cost_efficiency": float(round(r['pred_cost'], 2)),
                "biodegradability": float(r['bio']),
                "recyclability": float(r['recycl']),
                "tensile_strength": float(r['tensile']),
                "suitability_score": float(round(final_score, 1)),
                "is_recommended": bool(final_score > 70)
            })

        final_ranking.sort(key=lambda x: x['suitability_score'], reverse=True)
        return final_ranking

model_manager = ModelManager()
