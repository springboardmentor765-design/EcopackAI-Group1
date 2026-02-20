import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Ecopack-dataset.csv")

# CO2 Chart
plt.figure()
plt.bar(df["Material"], df["CO2_Emission_Score"])
plt.xticks(rotation=45)
plt.title("CO2 Emissions by Material")
plt.tight_layout()
plt.savefig("static/co2.png")

# Cost Chart
plt.figure()
plt.bar(df["Material"], df["Cost_per_kg"])
plt.xticks(rotation=45)
plt.title("Cost per Kg by Material")
plt.tight_layout()
plt.savefig("static/cost.png")
