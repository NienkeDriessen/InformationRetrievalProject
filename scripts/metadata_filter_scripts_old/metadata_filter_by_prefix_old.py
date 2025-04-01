import pandas as pd



# Load the global metadata CSV file
metadata_path = "../../metadata/metadata.csv"
df = pd.read_csv(metadata_path)

# Define the paths we want to filter
original_prefix = "original/images/OpenImages/"
modified_prefix = "inpainting/OpenImages/Kandinsky_2_2/"

# Filter rows where 'image_path' contains the specified prefixes
filtered_df = df[df["image_path"].str.startswith((original_prefix, modified_prefix))]

# Save the filtered data_tar_files to a new CSV file
output_path = "../../data_openImages/Original/metadata_Original.csv"
filtered_df.to_csv(output_path, index=False)

print(f"Filtered metadata saved to {output_path}")
