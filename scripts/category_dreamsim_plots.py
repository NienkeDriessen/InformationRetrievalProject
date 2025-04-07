import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
metadata_path = "../metadata/metadata_OpenImages.csv"
plots_folder = "../data_plots/"

# Ensure the plots folder exists
os.makedirs(plots_folder, exist_ok=True)

# Load the CSV file
df = pd.read_csv(metadata_path)

# Define categorical variables
categories = [
    "mse_rgb_category", "ssim_rgb_category", "ratio_rgb_category", "ratio_category",
    "change_location_category", "sem_mag_category", "size_mag_category"
]

# Check for missing categories
missing_categories = [cat for cat in categories if cat not in df.columns]
if missing_categories:
    raise ValueError(f"Missing expected category columns: {missing_categories}")

# Plot DreamSim values against each categorical variable
for category in categories:
    plt.figure(figsize=(8, 6))
    df.boxplot(column="dreamsim", by=category)
    plt.title(f"DreamSim vs {category}")
    plt.suptitle("")  # Remove automatic suptitle
    plt.xlabel(category)
    plt.ylabel("DreamSim")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(plots_folder, f"dreamsim_vs_{category}.png"))
    plt.close()

print("Plots saved in", plots_folder)
