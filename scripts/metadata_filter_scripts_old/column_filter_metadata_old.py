import pandas as pd

# Define the metadata file path
metadata_path = "../../metadata/metadata_OpenImages.csv"

# Load the CSV file
df = pd.read_csv(metadata_path)

# Define the columns to extract
columns_to_keep = [
    "image_id", "entities", "class", "label", "method", "model", "perturbed_mask_name",
    "sem_magnitude", "quality_flag", "cap2_img2", "direct_sim", "language_model",
    "dreamsim", "lpips_score", "sen_sim", "clip_sim", "mse_rgb", "mse_gray",
    "ssim_rgb", "ssim_gray", "ratio_rgb", "ratio_gray", "largest_component_size_gray",
    "largest_component_size_rgb", "cc_cluters_rgb", "cc_clusters_gray",
    "cluster_dist_rgb", "change_location_category", "sem_mag_category",
    "size_mag_category"
]

# Filter rows where image_id starts with a certain ID
filtered_df = df[df["image_id"].astype(str).str.startswith("ca4a95572a412288")]

# Convert to dictionary format
filtered_dict = filtered_df[columns_to_keep].to_dict(orient="records")

# Print the dictionary output
for entry in filtered_dict:
    print(entry)
