import os.path

import clip
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from load_data import save_embeddings

metadata_path = "ellende69.csv"


def extract_data_from_metadata(metadata: pd.DataFrame, save_path: str = '') -> (dict, list[str], list[float]):
    """
    Generate the mapping of alternate image index to original index.
    :param metadata: metadata dataframe
    :return: dataframe with 'index' and 'og_image' columns
    """
    semantic_metrics = ['dreamsim', 'mse_rgb', 'mse_gray', 'ssim_rgb', 'ssim_gray']
    # df = metadata[['index', 'image_id', 'image_path', 'ratio_category'] + semantic_metrics].copy()
    df = metadata.copy()

    # Track min and max values for specified metrics
    metric_values = {metric: [] for metric in semantic_metrics}

    # List to store selected original images with unique entity sets
    selected_originals = []
    unique_entity_sets = set()
    original_to_altered = {}
    alter_og = []

    # Filter metadata to remove all real image records w/out entities
    mask = ((df['label'] == 'real') & df['entities'].notna()) | (df['label'] == 'fake')
    df = df[mask]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        index = row['index']
        dreamsim = row['dreamsim']
        ratio = row['ratio_category']

        # Extract original ID (before any underscores if applicable)
        original_id = row['image_id'].split("_")[0]
        original_index = df.loc[df['image_id'] == original_id, 'index'].values[0] if not df[
            df['image_id'] == original_id].empty else None
        if original_index is None:
            # print(f"Original ID not found for {row['image_id']}")
            continue

        is_original = (row['label'] == 'real')
        if original_index not in original_to_altered:
            original_to_altered[original_index] = []
        if (not is_original) and dreamsim >= 0.13 and type(ratio) is str:
            altered_data = {
                "altered_index": index,
                "dreamsim": dreamsim,
                "mse_rgb": row["mse_rgb"],
                "mse_gray": row["mse_gray"],
                "ssim_rgb": row["ssim_rgb"],
                "ssim_gray": row["ssim_gray"],
            }
            original_to_altered[original_index].append(altered_data)
            alter_og.append([index, original_index])

            # Track values for distribution analysis
            for metric in semantic_metrics:
                metric_values[metric].append(row[metric])

    # Filter out original images with less than altered images
    filtered_originals = {k: v for k, v in original_to_altered.items() if len(v) >= 3}

    for og_index, altered_list in filtered_originals.items():
        # Check if the entity set is unique
        entities = df[df["index"] == og_index]["entities"].dropna().values.tolist()
        entities = entities[0].replace("'", "").replace("[", "").replace("]", "").split(", ")
        # Convert to a sorted tuple to ensure uniqueness
        entity_tuple = tuple(sorted(entities))

        # Check for duplicates and only add unique entity sets
        if entity_tuple and entity_tuple not in unique_entity_sets:
            unique_entity_sets.add(entity_tuple)
            selected_originals.append({
                "index": og_index,
                "entities": entities,
                "num_altered": len(altered_list),
                "altered_indices": [alt["altered_index"] for alt in altered_list]  # List of altered image IDs
            })
    return selected_originals, alter_og, semantic_metrics, metric_values


# def altered_to_og(selected_originals: list[dict]) -> pd.DataFrame:
#     """
#     Get the mapping of alternate image index to original index.
#     :param selected_originals: list of selected original images
#     :return: dataframe with 'index' and 'og_image' columns
#     """
#     image_id = []
#     og_image = []
#     for item in selected_originals:
#         for i in item['altered_indices']:
#             image_id.append(np.float64(item['index']))
#             og_image.append(np.float64(i))
#     return pd.DataFrame({'index': image_id, 'og_image': og_image})


def metric_value_analysis(metric_values: dict[str, list[float]]) -> None:
    """
    Print the min and max values for each metric.
    :param metric_values: dictionary of metric values
    """
    for metric, values in metric_values.items():
        print(f"{metric} - min: {min(values)}, max: {max(values)}")


def gen_query_embeddings(entity_list: list, model, preprocess) -> list[torch.Tensor]:
    """
    Generate embeddings for a list of entity lists. Finds embedding for individual entities and uses mean pooling
    to combine them.
    :param entity_list: list of entity lists
    :param model: CLIP model
    :param preprocess: CLIP preprocess
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings_list = []
    with torch.no_grad():
        for entities in tqdm(entity_list, total=len(entity_list)):
            ent = entities[0].replace("'", "").replace("[", "").replace("]", "").split(", ")
            tokens = clip.tokenize(ent).to(device)  # tokenise
            embeddings = model.encode_text(tokens)  # encode text
            embeddings /= embeddings.norm(dim=-1, keepdim=True)  # normalise
            combined = embeddings.mean(dim=0)  # average
            combined /= combined.norm(dim=-1, keepdim=True)  # normalise
            embeddings_list.append(combined)
    return embeddings_list


def generate_queries(metadata: pd.DataFrame, model, preprocess, save_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Generate queries based on selected original images and save to a CSV file.
    :param metadata: metadata dataframe
    :param model: CLIP model
    :param preprocess: CLIP preprocess
    :param save_path: path to save the queries
    """

    selected_originals, alt_og, semantic_metrics, metric_values = extract_data_from_metadata(metadata)

    # convert to DF
    df_selected = pd.DataFrame(selected_originals)
    df_selected.to_csv(os.path.join(save_path, 'query_information.csv'), index=False)

    # print the min and max values for each metric
    metric_value_analysis(metric_values)

    # og_to_alt into metadata - necessary for evaluation
    # ato = altered_to_og(original_to_altered)
    ato = pd.DataFrame(alt_og, columns=['index', 'og_image'])
    mdata = metadata[['index', 'image_id', 'image_path', 'label', 'entities', 'ratio_category'] + semantic_metrics].copy()
    mdata = mdata.merge(ato, left_on='index', right_on='index', how='left')
    mdata.to_csv(os.path.join(save_path, 'metadata_w_og_images.csv'), index=False)

    # get embeddings
    entities = df_selected['entities'].tolist()
    embeddings = gen_query_embeddings(entities, model, preprocess)
    indexes = df_selected['index'].tolist()
    save_embeddings(os.path.join(save_path, 'query_embeddings.h5'), embeddings, indexes)

    return mdata, df_selected

# -------------------------------------------------------------
metadata = pd.read_csv(metadata_path)
model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
save_path = '../data/'

mdata, df_selected = generate_queries(metadata, model, preprocess, save_path)



