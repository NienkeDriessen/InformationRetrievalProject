import json
import clip
import torch
import os
import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity
from load_data import save_embeddings
import torch.nn.functional as F
from tqdm import tqdm


def gen_entity_embeddings(mdata: pd.DataFrame, save_path: str, clip_model, clip_preprocess) -> {}:
    """
    Generate embeddings for entities of real images and save this to h5 file
    :param mdata: DataFrame containing the metadata of only real images
    :param save_path: Path where the embeddings will be saved.
    :param clip_model: CLIP model to use.
    :param clip_preprocess: CLIP preprocess to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = []
    image_paths = []
    image_indices = []
    omdata = mdata.loc[mdata['label'] == 'real'].copy()
    with torch.no_grad():
        for index, row in omdata.iterrows():
            # create embedding for the list of entities by individually getting embeddings,
            # then normalising and averaging
            string_list = row['entities'].replace("'", "").replace("[", "").replace("]", "").split(", ")
            tokens = clip.tokenize(string_list).to(device)
            entity_embeddings = clip_model.encode_text(tokens)  # shape: (N, D)
            entity_embeddings /= entity_embeddings.norm(dim=-1, keepdim=True)
            combined = entity_embeddings.mean(dim=0)  # shape: (D,)
            combined /= combined.norm()
            embeddings.append(combined)
            image_paths.append(row['image_path'])
            image_indices.append(row['index'])
    save_embeddings(os.path.join(save_path, 'entity_embeddings.h5'), embeddings, image_indices)
    return {'index': image_indices, 'image_path': image_paths, 'embeddings': embeddings}
    # { indices: [], paths: [], embeddings: [] }


def gen_ground_truth(mdata: pd.DataFrame, queries: pd.DataFrame, embeddings, model, preprocess, save_path: str,
                     ps_rel: int = 10) -> dict:
    query_dict = {}
    real_mdata = mdata.loc[mdata['label'] == 'real'].copy()  # only keep real images for ground truth
    with torch.no_grad():
        # stack embeddings for efficiency
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_list = [torch.tensor(embeddings[key][:]) for key in embeddings.keys()]
        s_embeddings = torch.stack(tensor_list)
        for index, row in tqdm(queries.iterrows()):
            ranking = real_mdata[['index', 'image_id', 'image_path']].copy()
            # get query embedding
            query_embedding = torch.Tensor(embeddings[row['index']])  # model.encode_text(clip.tokenize(row['query']).to(device))
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)  # normalise

            # record and sort distances
            ranking['cosine_distance'] = F.cosine_similarity(query_embedding, s_embeddings, dim=1)
            ranking = ranking.sort_values(by='cosine_distance', ascending=False)

            # pseudo-relevance feedback for binary scores
            rel_scores = np.zeros(len(ranking))
            rel_scores[:ps_rel] = 1
            ranking['ps_rel_feedback'] = rel_scores

            # record
            query_dict[row['index']] = ranking.to_dict(orient='records')
    # save to json
    with open(os.path.join(save_path, "queries_w_ground_truth.json"), "w") as outfile:
        json.dump(query_dict, outfile, default=lambda df: json.loads(df.to_json()))
    return query_dict
