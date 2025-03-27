import pandas as pd
import random
import ast  # To safely convert string representations of lists into actual lists

# Define the metadata file path
metadata_path = "../metadata/metadata_OpenImages.csv"

# Load the CSV file
df = pd.read_csv(metadata_path)

# Extract unique words from "entities" column
entities_set = set()
for entry in df["entities"].dropna():
    if isinstance(entry, str) and entry.startswith("["):
        try:
            words = ast.literal_eval(entry)  # Convert string-list to actual list
            for word in words:
                entities_set.update(word.split())  # Split multi-word labels
        except:
            continue  # Skip problematic entries
    else:
        entities_set.update(entry.split())  # Normal split for single words

# Extract unique words from "perturbed_mask_name"
perturbed_words = set()
for phrase in df["perturbed_mask_name"].dropna().unique():
    perturbed_words.update(phrase.split())  # Split multi-word phrases

# Convert sets to lists for random sampling
entities_list = list(entities_set)
perturbed_words_list = list(perturbed_words)

# Generate 30 subsets of 4 words (2 from each category)
subsets = []
for _ in range(30):
    if len(entities_list) >= 2 and len(perturbed_words_list) >= 2:
        subset = random.sample(entities_list, 2) + random.sample(perturbed_words_list, 2)
        random.shuffle(subset)  # Shuffle mix of entities & mask words
        subsets.append(subset)

# Print the subsets
for i, subset in enumerate(subsets, start=1):
    print(f"Subset {i}: {subset}")
