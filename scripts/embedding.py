"""
Module containing functionality regarding embeddings.
"""
import numpy as np
import pandas as pd
import torch

from load_data import find, find_index_from_image_id
import PIL.Image as Image


def generate_image_embeddings(imgs_path: str, metadata: pd.DataFrame, image_ids_to_real_indices: dict, model, preprocess, device) -> ([torch.Tensor], [str]):
    """
    Generate embeddings for images.

    :param imgs_path: Path to images folder.
    :param metadata: Metadata dataframe.
    :param image_ids_to_real_indices: Map of image ids to the real image indices they belong to.
    :param model: Embedding model.
    :param preprocess: Preprocessing function.
    :param device: Torch device.
    :return: Embeddings and image indices corresponding to images used to generate embeddings.
    """
    fake_images = []
    metallic = metadata[metadata['label'] == 'real']
    query_image_ids = [id for id in image_ids_to_real_indices.keys()]

    # Image ids of original images that also have a query associated to it
    real_images = [img_id for img_id in np.asarray(metallic['image_id']) if img_id in query_image_ids]
    altered_img_paths = []
    image_paths = []
    image_indices = []

    # Find altered images for each original image
    for i in range(len(real_images)):
        meta = metadata[metadata['image_id'].str.contains(real_images[i])]  # multiple images
        real_index = image_ids_to_real_indices[real_images[i]]
        image_indices.append(real_index)
        real_images[i] = real_images[i] + '.jpg'
        image_paths.append(find(real_images[i], imgs_path))

        alter1 = []
        alter1path = []
        metal = meta[meta['label'] == 'fake']

        for row in metal.itertuples():
            altered_id = row.image_id
            altered_index = row.index  # This refers to the 'index' column
            image_indices.append(altered_index)
            alter1.append(altered_id + '.png')
            alter1path.append(find(altered_id + '.png', imgs_path))
            image_paths.append(find(altered_id + '.png', imgs_path))
        fake_images.append(alter1)
        altered_img_paths.append(alter1path)

    # Generate embeddings for all images
    output_embeddings = []
    for i in range(len(image_paths)):
        if image_paths[i] is not None:
            path = str(image_paths[i])
            image = preprocess(Image.open(str(path))).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                output_embeddings.append(image_features)
        else:
            image_indices[i] = None
    image_indices = [int(i) for i in image_indices if i is not None]
    # image_paths = [path for path in image_paths if path is not None]
    print('embeddings shape: ', len(output_embeddings))
    return output_embeddings, image_indices
