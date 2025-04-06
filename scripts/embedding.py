"""
Module containing functionality regarding embeddings.
"""
import numpy as np
import pandas as pd
import torch

from load_data import find
import PIL.Image as Image


def generate_image_embeddings(imgs_path: str, metadata: pd.DataFrame, image_ids: [str], model, preprocess, device) -> ([torch.Tensor], [str]):
    """
    Generate embeddings for images.

    :param imgs_path: Path to images folder.
    :param metadata: Metadata dataframe.
    :param image_ids: List of image ids to load (all images we have queries for).
    :param model: Embedding model.
    :param preprocess: Preprocessing function.
    :param device: Torch device.
    :return: Embeddings and image paths corresponding to images used to generate embeddings.
    """
    fake_images = []
    metallic = metadata[metadata['label'] == 'real']

    # Image ids of original images that also have a query associated to it
    real_images = [img_id for img_id in np.asarray(metallic['image_id']) if img_id in image_ids]
    og_img_paths = []
    altered_img_paths = []
    image_paths = []

    mdata = metadata[['image_path', 'image_id', 'ratio_category', 'caption_embedding']].copy().set_index('image_path')

    # Find altered images for each original image
    for i in range(len(real_images)):
        meta = metadata[metadata['image_id'].str.contains(real_images[i])]  # multiple images
        meta = meta[meta["image_id"].isin(image_ids)]  # ????
        real_images[i] = real_images[i] + '.jpg'
        og_img_paths.append(find(real_images[i], imgs_path))
        image_paths.append(find(real_images[i], imgs_path))
        alter1 = []
        alter1path = []
        metal = meta[meta['label'] == 'fake']
        for j in metal['image_id']:
            alter1.append(j + '.png')
            alter1path.append(find(j + '.png', imgs_path))
            image_paths.append(find(j + '.png', imgs_path))
        fake_images.append(alter1)
        altered_img_paths.append(alter1path)

    image_paths[:] = [x for x in image_paths if x]

    linked_images = [{'Real_image': x, 'Altered_Image': y} for x, y in zip(real_images, fake_images)]
    linked_image_paths = [{'Real_image_path': x, 'Altered_Image_path': y} for x, y in
                          zip(og_img_paths, altered_img_paths)]



    # Generate embeddings for all images
    output_embeddings = []
    dict_embed = {}
    for i in range(len(image_paths)):
        if image_paths[i] != '[None]':
            image = preprocess(Image.open(str(image_paths[i]))).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                output_embeddings.append(image_features)
                dict_embed.update({'image': image_paths[i], 'embeddings': image_features})

    return output_embeddings, image_paths, mdata
