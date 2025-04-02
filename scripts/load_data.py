import os

import PIL.Image as Image
import clip
import h5py
import pandas as pd
import torch


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load metadata from CSV file.

    :param metadata_path: Path to CSV file.
    :return: DataFrame with metadata
    """
    return pd.read_csv(metadata_path)


def load_images_and_generate_embeddings(imgs_path: str, metadata: pd.DataFrame, embeddings_path: str, embedding_model) -> dict:
    """
    Load the images, filter only the images we want to use based on metadata.
    Also generates embeddings for these images and saved them to a h5

    :param imgs_path: Path to images folder.
    :param metadata: Metadata dataframe.
    :param embeddings_path: Path to embeddings folder.
    :param embedding_model: Embedding model.
    :return: Dictionary with image paths mapped to embeddings.
    """
    # TODO load images and filter only the ones we want based on metadata

    # TODO for
    for i in range(len(metadata)):
        metadata.iloc[i, 0] = metadata.iloc[i, 0] + '.png'
        for k in range(len(metadata[i, 4])):
            metadata.iloc[i, 4][k] = metadata.iloc[i, 4][k] + '.png'

    og_img_paths = []
    for i in range(len(metadata)):
        img_path = find(metadata.iloc[i, 0], imgs_path)
        og_img_paths.append(img_path)

    altered_img_paths = []
    for i in range(len(metadata)):
        for k in range(len(metadata[i, 4])):
            img_path = find(metadata.iloc[i, 4][k], imgs_path)
            altered_img_paths.append(img_path)

    image_path = og_img_paths + altered_img_paths
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    output_embeddings = []
    # generate embeddings for all sampled images
    for i in range(len(image_path)):
        image = preprocess(Image.open(str(image_path[i])).unsqueeze(0).to(device))
        with torch.no_grad():
            image_features = embedding_model.encode_image(image)
            output_embeddings.append(image_features)

    # Save embeddings to h5 file
    with h5py.File(embeddings_path, 'w') as f:
        f.create_dataset('img_path', data=image_path)  # String paths
        f.create_dataset('embeddings', data=output_embeddings)


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
    Load query file from string path, as well as a list of all images that are in the query file

    :param query_path: Path to query file.
    :return: Dataframe consisting of {original_id: {keywords,query,num_altered,altered_ids}}.
    :return: image_list containing all original image ids as well as their altered image ids
    """
    # If there is no query file yet, call the queries_based_on_random words to sub select valid original images to base queries on
    # Add an AI generated or in our case manually created queries column called "query"

    # Load csv dataframe
    query_df = pd.read_csv(query_path)

    # get image_list, which is a list of all strings in the id column, and all sub strings in the arrays in the altered_ids column