import pandas as pd
import numpy as np
import json
import os

"""
To do:
x ground truth generation (+ save i think)
x nDCG
x relative delta
- check whether ground truth list needs to be altered (Invisible Relevance Bias paper)

Other notes: 

Thought of relevance score = distance between query and image embeddings
Actual relevance = distance between image and caption embeddings

> captions are the same for the original and AI images ->> same relevance score
> so the rankings should ideally be original1, altered1, b_altered1, original2, altered2, b_altered2, ...

When assessing one corpus (e.g.,  human-written), documents from the other (e.g., LLM-generated) 
are treated as non-relevant, though the original mixed-source ranking order is maintained.
This approach allows us to independently assess the performance of IR models on each corpus source.
"""


def ndcg_score(ground_truth: pd.DataFrame, retrieved_list: pd.DataFrame, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at K. By choosing ground truth and retrieved list, we can
    compute for different corpora.
    nDCG = DCG / iDCG
    (i)DCG = sum^k_{i=1}(relevance_score_i / log2(rank_i + 1))
    :param ground_truth: ground truth ranking with relevance scores - [index=image_path, relevance_score]
    :param retrieved_list: graded ranking with relevance scores - [index=image_path, relevance_score]
    :param k:
    :return:
    """
    # get ideal DCG for ground truth
    log = np.log2(np.arange(2, k + 2))
    idcg = (np.array(ground_truth[:k]['relevance_score']) / log).sum()  # TODO: check slice is correct
    dcg = (np.array(retrieved_list[:k]['relevance_score']) / log).sum()  # TODO: check slice is correct + change to use right relevance scores
    return dcg / idcg if idcg > 0 else 0


def relative_d_score(ndcg_c1: float, ndcg_c2: float) -> float:
    """
    Compute relative delta from the paper. This is the difference between the nDCG scores of the two corpora.
    :param ndcg_c1: nDCG score for corpus 1
    :param ndcg_c2: nDCG score for corpus 2
    :return: relative delta
    """
    return 2 * (ndcg_c1 - ndcg_c2) / (ndcg_c1 + ndcg_c2) if (ndcg_c1 + ndcg_c2) > 0 else 0


def filter_relevance_labels(corpus: str, ground_truth: pd.DataFrame, retrieved_list: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Set relevance labels for particular corpus, setting others to irrelevant (0).
    :corpus: corpus to filter for (e.g., 'og', 'c1', 'c2', 'c3')
    :param ground_truth: ground truth of original images
    :param retrieved_list: retrieved list
    :return:
    """
    retrieved_list_local = retrieved_list.copy().reset_index(drop=False)
    if corpus == 'og':
        return retrieved_list_local[['image_path']].merge(ground_truth, on='image_path', how='left', suffixes=('', '_target')).fillna(0)
    else:
        prediction = retrieved_list_local[['image_path']]
        prediction = prediction.merge(metadata[['image_path', 'category', 'og_image']], on='image_path', how='left')
        prediction.loc[prediction['category'] == 'real', 'og_image'] = prediction['image_path']
        prediction = prediction.merge(ground_truth, left_on='og_image', right_on='image_path', how='left', suffixes=('', '_target'))
        prediction = prediction.rename(columns={'relevance_score_target': 'relevance_score'})
        if corpus == 'all':
            return prediction
        else:
            # replace relevance score with 0 where category != corpus
            prediction.loc[prediction['category'] != corpus, 'relevance_score'] = 0
            prediction = prediction[['image_path', 'relevance_score']]
            return prediction


def evaluate_query(ground_truth: pd.DataFrame, retrieved_list: pd.DataFrame, ks: [int], metadata: pd.DataFrame):
    """
    Compute the different nDCG@k scores for the different corpora. This function is for one query.
    :param ground_truth: dataframe of ground truth relevance scores - [image_path, relevance_score]
    :param retrieved_list: retrieved document dataframe - [image_path, relevance_score]
    :param ks: value of k for nDCG@k
    :param metadata: Metadata dataframe for finding the original image path for altered images.
    :return: dictionary for nDCG and relative delta scores for each corpus
    """
    # TODO: assign relevance labels to altered images
    ndcg_scores = {}
    for k in ks:
        # separate ranking list for different corpora
        prediction = filter_relevance_labels('all', ground_truth, retrieved_list, metadata)
        ndcg_all = ndcg_score(ground_truth, prediction, k)
        prediction = filter_relevance_labels('og', ground_truth, retrieved_list, metadata)
        ndcg_og = ndcg_score(ground_truth, prediction, k)
        prediction = filter_relevance_labels('c1', ground_truth, retrieved_list, metadata)
        ndcg_c1 = ndcg_score(ground_truth, prediction, k)
        prediction = filter_relevance_labels('c2', ground_truth, retrieved_list, metadata)
        ndcg_c2 = ndcg_score(ground_truth, prediction, k)
        prediction = filter_relevance_labels('c3', ground_truth, retrieved_list, metadata)
        ndcg_c3 = ndcg_score(ground_truth, prediction, k)
        ndcg_scores[str(k)] = {
            'ndcg_all': ndcg_all,
            'ndcg_og': ndcg_og,
            'ndcg_c1': ndcg_c1,
            'ndcg_c2': ndcg_c2,
            'ndcg_c3': ndcg_c3,
            'relD_og_c1': relative_d_score(ndcg_og, ndcg_c1),
            'relD_og_c2': relative_d_score(ndcg_og, ndcg_c2),
            'relD_og_c3': relative_d_score(ndcg_og, ndcg_c3),
            'relD_c1_c2': relative_d_score(ndcg_c1, ndcg_c2),
            'relD_c1_c3': relative_d_score(ndcg_c1, ndcg_c3),
            'relD_c2_c3': relative_d_score(ndcg_c2, ndcg_c3),
        }
    return ndcg_scores


def evaluate_all_queries(queries: [str], ground_truth: dict, retrieved_list: dict, ks: [int],
                         metadata: pd.DataFrame, save_folder: str = '') -> dict:
    """
    Compute the different nDCG@k scores for the different corpora. This function is for all queries.
    :param queries: list of queries
    :param ground_truth: ordered ground truth list - [image_path, relevance_score]
    :param retrieved_list: retrieved document list (ordered) - pd.DataFrame[index=image_path, relevance_score]
    :param k: value of k for nDCG@k
    :param metadata: Metadata dataframe for finding the original image path for altered images.
    :return: dictionary for nDCG and relative delta scores for each corpus
    """
    # get all scores
    all_scores = [evaluate_query(ground_truth[q]['ranked_list'], retrieved_list[q]['ranked_list'], ks, metadata)
                  for q in queries]
    # get average scores
    avg_scores = {}
    # ['ndcg_og', 'ndcg_c1', 'ndcg_c2', 'ndcg_c3', 'relD_og_c1', 'relD_og_c2', 'relD_og_c3', 'relD_c1_c2',
    # 'relD_c1_c3', 'relD_c2_c3']
    for k in ks:
        avg_scores[str(k)] = ({
            'ndcg_all': np.mean([s[str(k)]['ndcg_all'] for s in all_scores]),
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
    with open(os.path.join(save_folder, "evaluation_results.json"), "w") as outfile:
        json.dump(avg_scores, outfile, default=lambda df: json.loads(df.to_json()))
    return avg_scores
