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
    real_images = []
    fake_images = []
    for i in range(len(metadata)):
        if metadata['label'] == 'real':
            real_images.append(metadata.iloc[i,0])

    for i in range(len(real_images)):
        meta = metadata[metadata['image_id'].str.contains(real_images[i])]
        metal = meta['label' == 'fake']
        fake_images.append(metal[i])
        
    linked_images = {
        'original': real_images,
        'fake': fake_images
    }

    meta_linked = pd.dataframe(linked_images)


    # TODO for
    for i in range(len(meta_linked)):
        meta_linked.iloc[i, 0] = meta_linked.iloc[i, 0] + '.png'
        for k in range(len(meta_linked[i, 1])):
            meta_linked.iloc[i, 1][k] = meta_linked.iloc[i, 1][k] + '.png'

    og_img_paths = []
    for i in range(len(meta_linked)):
        img_path = find(meta_linked.iloc[i, 0], imgs_path)
        og_img_paths.append(img_path)

    altered_img_paths = []
    for i in range(len(meta_linked)):
        for k in range(len(meta_linked[i, 1])):
            img_path = find(meta_linked.iloc[i, 1][k], imgs_path)
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
