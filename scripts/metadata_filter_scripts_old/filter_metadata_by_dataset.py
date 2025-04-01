import pandas as pd

# This is to select only one of the datasets in the metadata

# Load the metadata CSV file
metadata_path = "../../metadata/metadata.csv"
df = pd.read_csv(metadata_path)

# Filter rows where 'dataset' is 'OpenImages'
df_filtered = df[df["dataset"] == "OpenImages"]

# Save the filtered DataFrame to a new CSV file
output_path = "../../metadata/metadata_OpenImages.csv"
df_filtered.to_csv(output_path, index=False)

print(f"Filtered metadata saved to {output_path}")