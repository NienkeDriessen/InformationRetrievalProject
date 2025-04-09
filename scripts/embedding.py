"""
Module containing functionality regarding embeddings.
"""
import os
import pandas as pd
import torch

import PIL.Image as Image


def generate_image_embeddings(imgs_path: str, metadata: pd.DataFrame, img_indices: [int], model, preprocess, device) -> ([torch.Tensor], [str]):
    """
    Generate embeddings for images.

    :param imgs_path: Path to images folder.
    :param metadata: Metadata dataframe.
    :param query_df: Query information dataframe.
    :param image_ids_to_real_indices: Map of image ids to the real image indices they belong to.
    :param model: Embedding model.
    :param preprocess: Preprocessing function.
    :param device: Torch device.
    :return: Embeddings and image indices corresponding to images used to generate embeddings.
    """
    output_embeddings = []
    for index in img_indices:
        row = metadata.loc[metadata['index'] == index].iloc[0]
        img_path = row['image_path']
        image_full_path = os.path.join(imgs_path, img_path)

        image = preprocess(Image.open(str(image_full_path))).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            output_embeddings.append(image_features)

    return output_embeddings, img_indices
