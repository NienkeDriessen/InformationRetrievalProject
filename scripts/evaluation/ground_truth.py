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
    # raise Exception("Debugging")/
    embeddings = []
    image_paths = []
    omdata = mdata.loc[mdata['label'] == 'real'].copy()
    # print(len(omdata))
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
            # print(combined)
            # raise Exception("Debugging")
    save_embeddings(os.path.join(save_path, 'entity_embeddings.h5'), embeddings, image_paths)
    return {'image_path': image_paths, 'embeddings': embeddings}  # { paths: [], embeddings: [] }


def gen_ground_truth(mdata: pd.DataFrame, queries: pd.DataFrame, embeddings, model, preprocess, save_path: str,
                     ps_rel: int = 10) -> dict:
    query_dict = {}
    real_mdata = mdata.loc[mdata['label'] == 'real'].copy()  # only keep real images for ground truth
    with torch.no_grad():
        # stack embeddings for efficiency
        # print(embeddings.keys())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_list = [torch.tensor(embeddings[key][:]) for key in embeddings.keys()]
        s_embeddings = torch.stack(tensor_list)
        # print(s_embeddings.shape)
        for index, row in tqdm(queries.iterrows()):
            # print(row)
            ranking = real_mdata[['image_id', 'image_path']].copy()
            # get query embedding
            query_embedding = model.encode_text(clip.tokenize(row['query']).to(device))
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)  # normalise
            # print(query_embedding.shape)
            # query_expanded = query_embedding.expand_as(s_embeddings)
            # record and sort distances
            ranking['cosine_distance'] = F.cosine_similarity(query_embedding, s_embeddings, dim=1)
            ranking = ranking.sort_values(by='cosine_distance', ascending=False)
            # pseudo-relevance feedback for binary scores
            rel_scores = np.zeros(len(ranking))
            rel_scores[:ps_rel] = 1
            ranking['ps_rel_feedback'] = rel_scores
            # record
            query_dict[row['id']] = ranking.to_dict(orient='records')
            # raise Exception("Debugging")
    # save to json
    with open(os.path.join(save_path, "queries_w_ground_truth.json"), "w") as outfile:
        json.dump(query_dict, outfile, default=lambda df: json.loads(df.to_json()))
    return query_dict


# def gen_ground_truth(queries: pd.DataFrame, metadata: pd.DataFrame, save_path: str,
#                      clip_model, clip_preprocess, ps_rel: int = None ) -> {}:
#     """
#     Generate ground truth for the dataset for the real images and save all to a JSON file. Assumes entity labels are
#     accurate (blame Semi-Truths otherwise).
#     :param queries: DataFrame containing the queries [image_path, query] where image path is the path of the image that
#     the query is based on.
#     :param metadata: DataFrame containing the metadata [image_path, original_image, entities, ...]
#     :param save_path: Path where the ground truth JSON file will be saved.
#     :param clip_model: CLIP model to use.
#     :param clip_preprocess: CLIP preprocess to use.
#     :param ps_rel: For pseudo-relevance feedback, how many top results are assumed to be relevant. If None, use cosine
#     similarity.
#     :return: Dictionary containing the queries and a dataframe with the ranked list of images and their relevance
#     labels.
#     """
#
#     mdata = metadata.loc[metadata['label'] == 'real'].copy()  # only keep real images for ground truth
#     mdata.dropna(axis=0, subset=['entities'], inplace=True)  # drop rows with no image path
#
#     ed = gen_entity_embeddings(mdata, save_path, clip_model, clip_preprocess)  # { paths: [], embeddings: [] }
#     query_dict = {}
#
#     ranking = mdata[['image_id', 'image_path']].copy().dropna(axis=0)
#     ranking['caption_embedding'] = ranking['image_path'].map(ed)
#
#     # for all queries, get the cosine similarity to rank for ground truth
#     for i, row in queries.iterrows():
#         embedding = clip_model.encode_text(clip.tokenize(row['query']).to(device))  # get embedding of query
#           # copy of mdata for ranking
#
#         ranking['query_embedding'] = embedding # TODO: check if this is the way
#         ranking['cosine_distance'] = cosine_similarity(ranking['caption_embedding'], embedding)
#         ranking = ranking.sort_values(by='cosine_distance', ascending=False)
#
#         if ps_rel is None:
#             # use cosine similarity as relevance score
#             ranked_list = ranking[['image_path', 'cosine_distance']].tolist()
#         else:
#             # pseudo relevance feedback: assume top ps_rel images are relevant/clicked
#             ranking['ps_rel_feedback'] = np.zeros(len(ranking))
#             ranking.iloc[:ps_rel]['ps_rel_feedback'] = 1
#             ranked_list = ranking[['image_path', 'ps_rel_feedback']].copy().set_index('image_path')
#
#         query_dict[row['id']] = {
#             'query': row['query'],
#             'ranked_list': ranked_list,  # DF[image_id, rel_score (cosine similarity / pseudo relevance)]
#         }
#     with open(os.path.join(save_path, "queries_w_ground_truth.json"), "w") as outfile:
#         json.dump(query_dict, outfile, default=lambda df: json.loads(df.to_json()))
#     return query_dict

"""
TODO:
- go over vectorisation in finding query embeddings and comparing to metadata ones >>> se below
- check rest
- try running

"""

"""
# EG how to use cosine similarity with PyTorch

import torch
import torch.nn.functional as F

# Reference embedding (1D tensor)
ref_embedding = torch.tensor([0.1, 0.2, 0.3])

# List of individual tensors
embedding_list = [
    torch.tensor([0.1, 0.2, 0.3]),
    torch.tensor([0.3, 0.2, 0.1]),
    torch.tensor([0.0, 0.1, 0.0])
]

# Stack into a single 2D tensor
embedding_tensor = torch.stack(embedding_list)  # shape: (n, dim)

# Normalize
ref_norm = F.normalize(ref_embedding, dim=0)
embedding_norm = F.normalize(embedding_tensor, dim=1)

# Cosine similarity: dot product between each row and ref
cosine_similarities = torch.matmul(embedding_norm, ref_norm)

print(cosine_similarities)


"""