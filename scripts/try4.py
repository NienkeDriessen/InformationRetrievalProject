import pandas as pd
import numpy as np
import h5py
import clip
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json

k = 10

"""
To do:
x ground truth generation (+save i think)
- evaluation metrics
    - average rank change (for fun)
    - R@k - recall @ k - weird met non-binary relevance maar anders gwn threshold
    - nDCG - general
    - relativeD - relative delta from paper
    
Thought of relevance score = distance between query and image embeddings
Actual relevance = distance between image and caption embeddings

> captions are the same for the original and AI images ->> same relevance score
> so the rankings should ideally be original1, altered1, b_altered1, original2, altered2, b_altered2, ...
> 

When assessing one corpus (e.g.,  human-written), documents from the other (e.g., LLM-generated) 
are treated as non-relevant, though the original mixed-source ranking order is maintained.
This approach allows us to independently assess the performance of IR models on each corpus source.
"""


def gen_ground_truth(queries: pd.DataFrame, metadata: pd.DataFrame, ps_rel: int = None) -> {}:
    """
    Generate ground truth for the dataset for the real images and save all to a JSON file. Assumes entity labels are
    accurate (blame Semi-Truths otherwise).
    :param queries: DataFrame containing the queries [image_path, query] where image path is the path of the image that
    the query is based on.
    :param metadata: DataFrame containing the metadata [image_path, original_image, entities, ...]
    :return: Dictionary containing the queries and their ranked list of images.
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
            ranked_list = ranking[['image_path', 'ps_rel_feedback']].tolist()

        query_dict[row['image_path']] = {
            'query': row['query'],
            'ranked_list': ranked_list,  # [image_id, rel_score (cosine similarity)]
        }
    with open("queries_w_ground_truth.json", "w") as outfile:
        json.dump(query_dict, outfile)
    return query_dict


def get_DCG_by_itself(relevance_scores: list, k: int) -> float:
    topk = np.array(relevance_scores[:k])
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(topk / discounts)


def ndcg_at_k(ground_truth, retrieved_list, k):
    """Compute Normalized Discounted Cumulative Gain (nDCG) at K."""
    # Generate ideal ranking (sorted by relevance)
    ideal_relevance = [1] * len(ground_truth)  # Ideal case: all relevant docs ranked perfectly
    ideal_dcg = get_DCG_by_itself(ideal_relevance, k)

    # Create relevance scores for retrieved documents (1 if in ground truth, else 0)
    retrieved_relevance = [1 if doc_id in ground_truth else 0 for doc_id, _ in retrieved_list]
    actual_dcg = get_DCG_by_itself(retrieved_relevance, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
