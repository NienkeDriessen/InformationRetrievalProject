# Information Retrieval Project

This repository contains the codebase for our research project on bias in image retrieval systems. Specifically, 
whether these systems are biased toward certain levels of AI-generated image alterations.

We work with the Semi-Truths dataset and analyze semantic / non-semantic image changes.

## Setup

### 1. Clone the repository:
```
git clone https://github.com/NienkeDriessen/InformationRetrievalProject.git

cd InformationRetrievalProject
```
### 2. Create + activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Dataset & Preprocessing

Download image metadata and tar files from the Semi-Truths dataset:

https://huggingface.co/datasets/semi-truths/Semi-Truths/tree/main/metadata

run the [tar_to_images.py](scripts/tar_to_images.py) script to get the images for a certain dataset.


### 4. Filter metadata for specific subsets:

To filter by dataset (e.g., "OpenImages"):
```
python scripts/metadata_filter_scripts_old/filter_metadata_by_dataset.py

```
To filter by alteration model (e.g., "StableDiffusionInpainting"):
```
python scripts/metadata_filter_scripts_old/filter_metadata_by_alteration_model.py
```

### 5. Query Generation
Generate image queries using [final_query_selection.py](scripts/final_query_selection.py)

After generation, manually add a new column named query between keywords and num_altered in the resulting CSV. This column should contain natural language queries based on image entities

### 6. Running the Main Pipeline

To run the full experiment pipeline and generate results:

```
python main.py
```
