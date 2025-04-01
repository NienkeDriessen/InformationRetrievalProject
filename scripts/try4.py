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
            'ranked_list': ranked_list,  # [image_id, rel_score (cosine similarity / pseudo relevance)]
        }
    with open("queries_w_ground_truth.json", "w") as outfile:
        json.dump(query_dict, outfile)
    return query_dict


def DCG_score(relevance_scores: list, k: int) -> float:
    topk = np.array(relevance_scores[:k])
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(topk / discounts)

def nDCG_score(ground_truth, retrieved_list, k):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at K. By choosing ground truth and retrieved list, we can
    compute for different corpora.
    nDCG = DCG / iDCG
    (i)DCG = sum^k_{i=1}(relevance_score_i / log2(rank_i + 1))
    :param ground_truth:
    :param retrieved_list:
    :param k:
    :return:
    """
    # get ideal DCG for ground truth
    log = np.log2(np.arange(2, k + 2))
    iDCG = (ground_truth[:k] / log).sum()  # TODO: check slice is correct
    dcg = (retrieved_list[:k] / log).sum()  # TODO: check slice is correct
    return dcg / iDCG if iDCG > 0 else 0


def relativeD_score(ndcg_c1: float, ndcg_c2: float) -> float:
    """
    Compute relative delta from the paper. This is the difference between the nDCG scores of the two corpora.
    :param ndcg_c1: nDCG score for corpus 1
    :param ndcg_c2: nDCG score for corpus 2
    :return: relative delta
    """
    return 2 * (ndcg_c1 - ndcg_c2) / (ndcg_c1 + ndcg_c2) if (ndcg_c1 + ndcg_c2) > 0 else 0


def evaluate_query(ground_truth, retrieved_list, k: [int], metadata: pd.DataFrame):
    """
    Compute the different nDCG@k scores for the different corpora. This function is for one query.
    :param ground_truth: 2D list of ground truth relevance scores - [image_path, relevance_score]
    :param retrieved_list: retrieved document list - [image_path, relevance_score]
    :param k: value of k for nDCG@k
    :param metadata: Metadata dataframe for finding the original image path for altered images.
    :return: dictionary for nDCG and relative delta scores for each corpus
    """
    # get ndcg for each k
    ndcg_scores = {}
    for k in k:
        # separate ranking list for different corpora
        ndcg_og = 0
        ndcg_c1 = 0
        ndcg_c2 = 0
        ndcg_c3 = 0
        ndcg_scores[str(k)] = {
            'ndcg_og': ndcg_og,
            'ndcg_c1': ndcg_c1,
            'ndcg_c2': ndcg_c2,
            'ndcg_c3': ndcg_c3,
            'relD_og_c1': relativeD_score(ndcg_og, ndcg_c1),
            'relD_og_c2': relativeD_score(ndcg_og, ndcg_c2),
            'relD_og_c3': relativeD_score(ndcg_og, ndcg_c3),
            'relD_c1_c2': relativeD_score(ndcg_c1, ndcg_c2),
            'relD_c1_c3': relativeD_score(ndcg_c1, ndcg_c3),
            'relD_c2_c3': relativeD_score(ndcg_c2, ndcg_c3),
        }


def evaluate_all_queries(queries: [str], ground_truth, retrieved_list, k: [int], metadata: pd.DataFrame):
    """
    Compute the different nDCG@k scores for the different corpora. This function is for all queries.
    :param queries: list of queries
    :param ground_truth: 2D list of ground truth relevance scores - [image_path, relevance_score]
    :param retrieved_list: retrieved document list - [image_path, relevance_score]
    :param k: value of k for nDCG@k
    :param metadata: Metadata dataframe for finding the original image path for altered images.
    :return: dictionary for nDCG and relative delta scores for each corpus
    """
    # get all scores
    all_scores = [evaluate_query(ground_truth[q], retrieved_list[q], k, metadata) for q in queries]

    # get average scores
    avg_scores = {}
    # ['ndcg_og', 'ndcg_c1', 'ndcg_c2', 'ndcg_c3', 'relD_og_c1', 'relD_og_c2', 'relD_og_c3', 'relD_c1_c2', 'relD_c1_c3', 'relD_c2_c3']
    for k in k:
        avg_scores[str(k)]({
            'ndcg_og': np.mean([s[str(k)]['ndcg_og'] for s in all_scores]),
            'ndcg_c1': np.mean([s[str(k)]['ndcg_c1'] for s in all_scores]),
            'ndcg_c2': np.mean([s[str(k)]['ndcg_c2'] for s in all_scores]),
            'ndcg_c3': np.mean([s[str(k)]['ndcg_c3'] for s in all_scores]),
            'relD_og_c1': np.mean([s[str(k)]['relD_og_c1'] for s in all_scores]),
            'relD_og_c2': np.mean([s[str(k)]['relD_og_c2'] for s in all_scores]),
            'relD_og_c3': np.mean([s[str(k)]['relD_og_c3'] for s in all_scores]),
            'relD_c1_c2': np.mean([s[str(k)]['relD_c1_c2'] for s in all_scores]),
            'relD_c1_c3': np.mean([s[str(k)]['relD_c1_c3'] for s in all_scores]),
            'relD_c2_c3': np.mean([s[str(k)]['relD_c2_c3'] for s in all_scores]),
        })
