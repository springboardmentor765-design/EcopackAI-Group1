from utils.model import model_manager
import pandas as pd

def test_scenario(name, inputs):
    with open("verify_log.txt", "a") as f:
        f.write(f"\n--- Testing Scenario: {name} ---\n")
        f.write(f"Inputs: {inputs}\n")
        try:
            results = model_manager.predict_all(inputs)
            if not results:
                f.write("NO RESULTS RETURNED.\n")
                return

            top = results[0]
            f.write(f"WINNER: {top['material_type']}\n")
            f.write(f"Score: {top['suitability_score']} (CO2: {top['predicted_co2']}, Cost: {top['predicted_cost_efficiency']})\n")
            
            # Show Top 3 for context
            f.write("Top 3:\n")
            for i, r in enumerate(results[:3]):
                 f.write(f"  {i+1}. {r['material_type']} (Score: {r['suitability_score']})\n")
                 
        except Exception as e:
            f.write(f"CRASH: {e}\n")

# Clear log
open("verify_log.txt", "w").close()

# Scenario 1: Heavy Industrial (Should be Corrugated Box or Aluminum)
test_scenario("Heavy Load", {
    "weight_capacity": 15.0, 
    "category": "Industrial",
    "fragility_score": 5,
    "shelf_life_days": 30
})

# Scenario 2: Light Cosmetic (Should be Mushroom or Bioplastic)
test_scenario("Light Cosmetic", {
    "weight_capacity": 0.2, 
    "category": "Cosmetics",
    "fragility_score": 3,
    "shelf_life_days": 60
})

# Scenario 3: Fragile Electronics (Should be Corrugated Box or Mushroom)
test_scenario("Fragile Electronics", {
    "weight_capacity": 3.0, 
    "category": "Electronics",
    "fragility_score": 9,
    "shelf_life_days": 365
})

# Scenario 4: Long Shelf Life Food (Should be Aluminum or Glass - Paper must fail)
test_scenario("Preserved Food", {
    "weight_capacity": 1.0, 
    "category": "Food",
    "fragility_score": 2,
    "shelf_life_days": 90
})

# Scenario 5: Mid-Range Standard (Should be Cardboard or Corrugated)
test_scenario("Standard Parcel", {
    "weight_capacity": 5.0, 
    "category": "General",
    "fragility_score": 4,
    "shelf_life_days": 15
})

# Scenario 6: Extremely Heavy Industrial (Should be Corrugated Box or Virgin Plastic)
test_scenario("Heavy Industrial Machine", {
    "weight_capacity": 25.0,
    "category": "Industrial",
    "fragility_score": 3,
    "shelf_life_days": 365
})

# Scenario 7: Eco-Friendly Takeout (Should be Bagasse or Starch)
test_scenario("Fast Food Container", {
    "weight_capacity": 0.3,
    "category": "Food",
    "fragility_score": 2,
    "shelf_life_days": 1
})

# Scenario 8: Luxury Perfume (Should be Glass or High-End Bio)
test_scenario("Luxury Cosmetic", {
    "weight_capacity": 0.5,
    "category": "Cosmetics",
    "fragility_score": 8,
    "shelf_life_days": 730
})

# Scenario 9: Cheap Bulk Textbooks (Should be Kraft Paper or Cardboard)
test_scenario("Books Shipment", {
    "weight_capacity": 10.0,
    "category": "General",
    "fragility_score": 2,
    "shelf_life_days": 30
})

