import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
metadata_path = "../metadata/processed_metadata_OpenImages.csv"
plots_folder = "../data_plots/"

if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Define a function to check if an image is original (based on fixed-length IDs)
def isOriginal(image_id):
    return len(image_id) == 16  # Assuming original IDs have exactly 16 characters


# Load the CSV file
df = pd.read_csv(metadata_path)
mask = ((df['label'] == 'real') & df['entities'].notna()) | (df['label'] == 'fake')
df = df[mask]

print(df.columns)
print(len(df))

# Dictionary to store original images and their altered versions
original_to_altered = {}

# Track min and max values for specified metrics
tracked_metrics = ["dreamsim", "mse_rgb", "mse_gray", "ssim_rgb", "ssim_gray"]
metric_values = {metric: [] for metric in tracked_metrics}

# List to store selected original images with unique entity sets
selected_originals = []
unique_entity_sets = set()

# Iterate over the dataset
for _, row in df.iterrows():
    image_id = row['image_id']
    dreamsim = row['dreamsim']  # DreamSim value
    ratio = row['ratio_category']
    index = row['index']
    # print(row)

    # Extract original ID (before any underscores if applicable)
    original_id = image_id.split("_")[0]
    original_index = df.loc[df['image_id'] == original_id, 'index'].values[0] if not df[df['image_id'] == original_id].empty else None
    if original_index is None:
        print(f"Original ID not found for {image_id}")
        continue
        # raise Exception(f"Original ID not found for {image_id}")

    is_original = isOriginal(image_id)  # Check if the image is original
    # Check if it's an original image
    # if is_original:
    if original_index not in original_to_altered:
        original_to_altered[original_index] = []  # Initialize list
        # First filter: Only add altered images with DreamSim ≥ 0.13
    if (not is_original) and dreamsim >= 0.13 and type(ratio) is str:
        print("not none", ratio)
        altered_data = {
            "altered_id": index,
            "dreamsim": dreamsim,
            "mse_rgb": row["mse_rgb"],
            "mse_gray": row["mse_gray"],
            "ssim_rgb": row["ssim_rgb"],
            "ssim_gray": row["ssim_gray"],
        }
        original_to_altered[original_index].append(altered_data)

        # Track values for distribution analysis
        for metric in tracked_metrics:
            metric_values[metric].append(row[metric])


# Second filter: Only keep original images that have at least 3 altered versions
filtered_originals = {k: v for k, v in original_to_altered.items() if len(v) >= 3}

# print(filtered_originals)
# raise Exception("stop here")

# Retrieve entity keywords for the final selection
for orig_id, altered_list in filtered_originals.items():
    entities = df[df["index"] == orig_id]["entities"].dropna().values

    # Convert to a sorted tuple to ensure uniqueness
    entity_tuple = tuple(sorted(entities))

    # Check for duplicates and only add unique entity sets
    if entity_tuple and entity_tuple not in unique_entity_sets:
        unique_entity_sets.add(entity_tuple)
        selected_originals.append({
            "index": orig_id,
            "keywords": entities.tolist(),
            "num_altered": len(altered_list),
            "altered_ids": [alt["altered_id"] for alt in altered_list]  # List of altered image IDs
        })

# Print the selected original images with keywords, number of altered versions, and altered IDs
print("\nSelected Original Images (At least 3 altered versions & DreamSim ≥ 0.13):")
for entry in selected_originals:
    print(entry["keywords"])


altered_to_og = {}
for row in selected_originals:
    for i in row["altered_ids"]:
        altered_to_og[i] = row["index"]
df = pd.DataFrame.from_dict(altered_to_og, orient='index', columns=['og_image'])
df.to_csv(plots_folder + "altered_to_og2.csv", index=True)

# Convert to DataFrame and save to CSV
df_selected = pd.DataFrame(selected_originals)
df_selected.to_csv(plots_folder + "ratio_queries.csv", index=False)

# Print the min and max values for each metric
print("\nMetric Value Ranges:")
for metric in tracked_metrics:
    if metric_values[metric]:  # Ensure there are values
        print(f"{metric}: Min = {min(metric_values[metric]):.4f}, Max = {max(metric_values[metric]):.4f}")

# Generate the plots for these categoric metrics: mse_rgb_category,ssim_rgb_category,ratio_rgb_category,ratio_category,change_location_category,sem_mag_category,size_mag_category,
# Each category has bin1 ... bin5

#
# # Generate box plots for the tracked metrics
# plt.figure(figsize=(10, 6))
# plt.boxplot([metric_values[m] for m in tracked_metrics], tick_labels=tracked_metrics, vert=True)  # Updated parameter
# plt.title("Distribution of DreamSim and Image Quality Metrics")
# plt.ylabel("Values")
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.savefig(plots_folder + "boxplot.png")
# print("Boxplot saved as 'boxplot.png'.")
#
# # Generate scatterplots
# scatter_metrics = ["mse_rgb", "mse_gray", "ssim_rgb", "ssim_gray"]
# plt.figure(figsize=(12, 10))
#
# for i, metric in enumerate(scatter_metrics, 1):
#     plt.subplot(2, 2, i)
#     plt.scatter(metric_values["dreamsim"], metric_values[metric], alpha=0.5)
#     plt.xlabel("DreamSim")
#     plt.ylabel(metric)
#     plt.title(f"DreamSim vs {metric}")
#     plt.grid(True, linestyle="--", alpha=0.7)
#
# plt.tight_layout()
# plt.savefig(plots_folder + "scatterplots.png")
# print("Scatterplots saved as 'scatterplots.png'.")
