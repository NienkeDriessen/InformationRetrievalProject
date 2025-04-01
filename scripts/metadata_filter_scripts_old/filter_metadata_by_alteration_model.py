import pandas as pd

# This script is for filtering the metadata to only the ones we need from a specific model that changed the images
# So I divide it by Kandinsky_2_2, OpenJourney, SDv4 etc..

# Define file paths
metadata_file = "../../metadata/metadata.csv"
output_file = "../../metadata/metadata_StableDiffusion_XL.csv"

# Load the metadata
metadata = pd.read_csv(metadata_file)

# Filter rows where 'model' column is 'OpenJourney' (or any other of the models)
kandinsky_metadata = metadata[metadata['model'] == 'StableDiffusion_XL']

# Save the filtered metadata to a new CSV file
kandinsky_metadata.to_csv(output_file, index=False)

print(f"Filtered metadata saved to {output_file}")
