"""
Module containing functionality to load data (e.g. metadata, embeddings...)
"""
import os
import h5py
import pandas as pd
import torch


def find(name, path) -> os.path:
    """
    Find a file by name and path.

    :param name: Name of file to find.
    :param path: Path to file to find.

    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load metadata from CSV file.

    :param metadata_path: Path to CSV file.
    :return: DataFrame with metadata
    """
    return pd.read_csv(metadata_path)


def save_embeddings(embeddings_path: str, embeddings: [torch.tensor], image_paths: [str]) -> None:
    """
    Save embeddings to a h5 file along with corresponding image paths.

    :param embeddings_path: Path to embeddings file.
    :param embeddings: List of embeddings as tensors.
    :param image_paths: List of image paths (corresponding to embeddings in same order).
    """
    with h5py.File(embeddings_path, 'w') as f:
        f.create_dataset('img_path', data=image_paths)  # String paths
        f.create_dataset('embeddings', data=embeddings)


def load_embeddings(embeddings_path: str) -> dict:
    """
    Load embeddings from h5 file.

    :param embeddings_path: Path to h5 file.
    :return: Dictionary with image paths mapped to embeddings.
    """
    embedding_dict = {}
    with h5py.File(embeddings_path, 'r') as f:
        img_paths = f['img_path'][:].astype(str)
        embeddings = f['embeddings'][:]

        for i in range(len(img_paths)):
            embedding_dict[img_paths[i]] = embeddings[i]

    return embedding_dict


def load_queries_and_image_ids(query_path: str):
    """
    Load query file from string path, as well as a list of all images that are in the query file.

    :param query_path: Path to query file.
    :return: DataFrame consisting of {original_id: {keywords, query, num_altered, altered_ids}}.
    :return: image_list containing all original image ids as well as their altered image ids.
    """
    # Load CSV into DataFrame
    query_df = pd.read_csv(query_path)

    # Ensure required columns exist
    # id,keywords,query,num_altered,altered_ids
    required_columns = {'id', 'keywords', 'query', 'num_altered', 'altered_ids'}
    if not required_columns.issubset(query_df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(query_df.columns)}")

    # Convert altered_ids column (assumed to be a string representation of lists) into actual lists
    query_df['altered_ids'] = query_df['altered_ids'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Compile a list of all image IDs (original and altered)
    image_list = set(query_df['id'].tolist())
    for altered in query_df['altered_ids']:  # Assuming altered_ids is a list of strings
        image_list.update(altered)

    return query_df, list(image_list)
