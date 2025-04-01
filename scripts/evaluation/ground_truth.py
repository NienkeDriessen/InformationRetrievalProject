import json
import clip
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def gen_ground_truth(queries: pd.DataFrame, metadata: pd.DataFrame, ps_rel: int = None) -> {}:
    """
    Generate ground truth for the dataset for the real images and save all to a JSON file. Assumes entity labels are
    accurate (blame Semi-Truths otherwise).
    :param queries: DataFrame containing the queries [image_path, query] where image path is the path of the image that
    the query is based on.
    :param metadata: DataFrame containing the metadata [image_path, original_image, entities, ...]
    :return: Dictionary containing the queries and a dataframe with the ranked list of images and their relevance labels.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # get semantic embedding of all captions and queries
    metadata['caption_embedding'] = model.encode_text(clip.tokenize(metadata['caption']).to(device))
    query_embeddings = model.encode_text(clip.tokenize(queries['query']).to(device))

    query_dict = {}

    # for all queries, get the cosine similarity to rank for ground truth
    for i, row in queries.iterrows():
        ranking = metadata.copy()['image_path', 'caption_embedding']
        ranking['query_embedding'] = row['embedding']

        ranking['cosine_distance'] = cosine_similarity(ranking['caption_embedding'], query_embeddings)
        ranking = ranking.sort_values(by='cosine_distance', ascending=False)

        if ps_rel is None:
            # use cosine similarity as relevance score
            ranked_list = ranking[['image_path', 'cosine_distance']].tolist()
        else:
            # pseudo relevance feedback: assume top ps_rel images are relevant/clicked
            ranking['ps_rel_feedback'] = np.zeros(len(ranking))
            ranking.iloc[:ps_rel]['ps_rel_feedback'] = 1
            ranked_list = ranking[['image_path', 'ps_rel_feedback']].copy().set_index('image_path')

        query_dict[row['image_path']] = {
            'query': row['query'],
            'ranked_list': ranked_list,  # DF[image_id, rel_score (cosine similarity / pseudo relevance)]
        }
    with open("queries_w_ground_truth.json", "w") as outfile:
        json.dump(query_dict, outfile)
    return query_dict
