import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
metadata_path = "../metadata/metadata_OpenImages.csv"
plots_folder = "../data_plots/"

# Load the CSV file
df = pd.read_csv(metadata_path)

# Define a function to check if an image is original (based on fixed-length IDs)
def is_original(image_id):
    return len(image_id) == 16  # Assuming original IDs have exactly 16 characters

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

    # Extract original ID (before any underscores if applicable)
    original_id = image_id.split("_")[0]

    # Check if it's an original image
    if is_original(original_id):
        if original_id not in original_to_altered:
            original_to_altered[original_id] = []  # Initialize list

        # First filter: Only add altered images with DreamSim ≥ 0.13
        if not is_original(image_id) and dreamsim >= 0.13:
            altered_data = {
                "altered_id": image_id,
                "dreamsim": dreamsim,
                "mse_rgb": row["mse_rgb"],
                "mse_gray": row["mse_gray"],
                "ssim_rgb": row["ssim_rgb"],
                "ssim_gray": row["ssim_gray"],
            }
            original_to_altered[original_id].append(altered_data)

            # Track values for distribution analysis
            for metric in tracked_metrics:
                metric_values[metric].append(row[metric])

# Second filter: Only keep original images that have at least 3 altered versions
filtered_originals = {k: v for k, v in original_to_altered.items() if len(v) >= 3}

# Retrieve entity keywords for the final selection
for orig_id, altered_list in filtered_originals.items():
    entities = df[df["image_id"] == orig_id]["entities"].dropna().values

    # Convert to a sorted tuple to ensure uniqueness
    entity_tuple = tuple(sorted(entities))

    # Check for duplicates and only add unique entity sets
    if entity_tuple and entity_tuple not in unique_entity_sets:
        unique_entity_sets.add(entity_tuple)
        selected_originals.append({
            "id": orig_id,
            "keywords": entities.tolist(),
            "num_altered": len(altered_list),
            "altered_ids": [alt["altered_id"] for alt in altered_list]  # List of altered image IDs
        })

# Print the selected original images with keywords, number of altered versions, and altered IDs
print("\nSelected Original Images (At least 3 altered versions & DreamSim ≥ 0.13):")
for entry in selected_originals:
    print(entry["keywords"])


# Convert to DataFrame and save to CSV
df_selected = pd.DataFrame(selected_originals)
df_selected.to_csv(plots_folder + "queries.csv", index=False)

# Print the min and max values for each metric
print("\nMetric Value Ranges:")
for metric in tracked_metrics:
    if metric_values[metric]:  # Ensure there are values
        print(f"{metric}: Min = {min(metric_values[metric]):.4f}, Max = {max(metric_values[metric]):.4f}")

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
