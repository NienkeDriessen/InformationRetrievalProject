import pandas as pd

# Define the metadata file path
metadata_path = "../metadata/metadata_OpenImages.csv"
output_path = "../metadata/metadata_OpenImages_sorted.csv"

# Load sorted data
df_sorted = pd.read_csv(output_path)

# Extract relevant columns
image_ids = df_sorted.iloc[:, 0]  # First column (image_id)
labels = df_sorted[['image_id', 'entities', 'sem_magnitude']]  # Extract relevant columns

# Dictionary to store semantic magnitude categories for each original image
original_image_semantic_categories = {}

for _, row in labels.iterrows():
    img_id = row['image_id']
    original_id = img_id.split("_")[0]  # Extract original ID
    sem_magnitude = row['sem_magnitude']

    if len(original_id) == 16:  # Ensure it's an original image ID
        if original_id not in original_image_semantic_categories:
            original_image_semantic_categories[original_id] = set()

        # Track which semantic categories exist for this original image
        original_image_semantic_categories[original_id].add(sem_magnitude)

# List to store qualifying image IDs and their labels
qualifying_images = []

for orig_id, sem_categories in original_image_semantic_categories.items():
    if {'small', 'medium', 'large'}.issubset(sem_categories):  # Check if all categories exist
        label = labels[labels['image_id'] == orig_id]['entities'].values[0]
        qualifying_images.append((orig_id, label))

# Print results
print("Images with small, medium, and large changes:")
print("------------------------------------------")
for img_id, label in qualifying_images:
    print(f"{img_id} | {label}")

print(original_image_semantic_categories['94e41c296bd7ffb3'])