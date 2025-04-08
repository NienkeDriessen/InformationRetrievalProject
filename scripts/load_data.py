"""
Module containing functionality to load data (e.g. metadata, embeddings...)
"""
import os
from typing import List

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


def find_index_from_image_id(image_id: str, metadata: pd.DataFrame) -> str:
    """
    Find the index of an image by its image id from a metadata dataframe.

    :param image_id: Image id to find.
    :param metadata: Metadata dataframe.
    :param column_name: Name of column containing image ids.
    :return: Index of image.
    """
    return metadata[metadata['image_id'] == image_id]['index'].iloc[0]


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load metadata from CSV file.

    :param metadata_path: Path to CSV file.
    :return: DataFrame with metadata
    """
    return pd.read_csv(metadata_path)


def save_metadata(metadata: pd.DataFrame, metadata_path: str) -> None:
    """
    Save metadata to CSV file with index.

    :param metadata: Metadata to save
    :param metadata_path: Path to CSV file.
    """
    metadata.to_csv(metadata_path, index=True, index_label='index')


def save_embeddings(embeddings_path: str, embeddings: [torch.tensor], image_indices: [str]) -> None:
    """
    Save embeddings to a h5 file along with corresponding image paths.

    :param embeddings_path: Path to embeddings file.
    :param embeddings: List of embeddings as tensors.
    :param image_indices: List of image indices (corresponding to embeddings in same order).
    """
    with h5py.File(embeddings_path, 'w') as f:
        f.create_dataset('img_index', data=image_indices)  # String paths
        f.create_dataset('embeddings', data=embeddings)


def load_embeddings(embeddings_path: str) -> dict:
    """
    Load embeddings from h5 file.

    :param embeddings_path: Path to h5 file.
    :return: Dictionary with image indices mapped to embeddings.
    """
    embedding_dict = {}
    with h5py.File(embeddings_path, 'r') as f:
        img_indices = f['img_index'][:].astype(str)
        embeddings = f['embeddings'][:]

        for i in range(len(img_indices)):
            embedding_dict[img_indices[i]] = embeddings[i]

    return embedding_dict


def load_queries_and_image_ids(query_path: str, img_metadata: pd.DataFrame) -> (List[str], List[str]):
    """
    Load query file from string path, as well as a list of all images that are in the query file.

    :param query_path: Path to query file.
    :param img_metadata: Metadata dataframe for images.
    :return: DataFrame consisting of {original_id: {keywords, query, num_altered, altered_ids}}.
    :return: image_list containing all original image ids as well as their altered image ids.
    """
    # Load CSV into DataFrame
    query_df = pd.read_csv(query_path, sep=';')
    query_df['index'] = query_df['id'].apply(lambda x: find_index_from_image_id(x, img_metadata))

    # Ensure required columns exist
    # id,keywords,query,num_altered,altered_ids
    required_columns = {'id', 'keywords', 'query', 'num_altered', 'altered_ids'}
    if not required_columns.issubset(query_df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(query_df.columns)}")

    # Convert altered_ids column (assumed to be a string representation of lists) into actual lists
    query_df['altered_ids'] = query_df['altered_ids'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    query_df['altered_indices'] = query_df['altered_ids'].apply(lambda x: [find_index_from_image_id(img_id, img_metadata) for img_id in x])

    # Compile a list of all image IDs (original and altered)
    image_list = set(query_df['id'].tolist())
    for altered in query_df['altered_ids']:  # Assuming altered_ids is a list of strings
        image_list.update(altered)

    return query_df, list(image_list)
