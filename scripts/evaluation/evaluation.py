import os.path

import pandas as pd
from pprint import pprint
import json

from evaluation.evaluation_metrics import evaluate_all_queries
from evaluation.distance_exploration import compute_relative_distances_per_bin, plot_relative_distances
from evaluation.reformatting import reformat_retrieval_results

"""
Main script for evaluating the retrieval results of the image retrieval system.

Expected format of retrieval results: JSON file with the following structure:
{ <image_path_from_query>: pd.DataFrame[image_path, relevance_score] }
"""


def evaluate(k_values: list[int], retrieval_results_path: str, metadata: pd.DataFrame, queries: pd.DataFrame, ground_truth: dict,
             save_folder: str):
    """
    Full evaluation of the retrieval results.
    :param k_values: values of k to compute scores for.
    :param retrieval_results_path: path to retrieval results JSON file.
    :param metadata: metadata for images.
    :param queries: queries for images.
    :param save_folder: path to folder where to save results.
    :return: None
    """
    # reformat_metadata(metadata)
    # ground_truth = gen_ground_truth(queries, metadata=metadata, ps_rel=10, save_path=save_folder)
    # print(f'Saved generated ground truth to {save_folder} as a JSON file.')
    # print('Structure of path:\n <image_path>: { query: str, ranked_list: pd.DataFrame[image_path, relevance_score] }')

    retrieval_results = json.load(open(retrieval_results_path, 'r'))
    # for k,v in retrieval_results.items():
    #     print(v)
    #     break
    # raise Exception("Debugging")
    retrieval_results = {k: reformat_retrieval_results(pd.DataFrame(v)) for k, v in retrieval_results.items()}
    metadata['ratio_category'].fillna('real', inplace=True)  # TODO: will not work for pandas 3.0
    # metadata.fillna({'og_image': ''}, inplace=True)  # TODO: check if this is the right way to do it
    # 'df.method({col: value}, inplace=True)'

    if not os.path.exists(os.path.join(save_folder, "evaluation_results.json")):
        evaluation_results = evaluate_all_queries(queries['index'].tolist(), ground_truth, retrieval_results, k_values,
                                                  metadata=metadata, save_folder=save_folder)
        print(f'Saved generated ground truth to {save_folder} as a JSON file.')
        print('Structure of path:\n { <k>: { ndcg_[all|c1|...|og]: float, relD_[og|c1|...]_[c1|...]: float } }')
        pprint(evaluation_results)

    relative_distances = compute_relative_distances_per_bin(['bin1', 'bin2', 'bin3', 'bin4', 'bin5'], retrieval_results,
                                                            metadata)
    print('Relative distances per bin:')
    pprint(relative_distances)

    plot_relative_distances(relative_distances, save_path=save_folder)

