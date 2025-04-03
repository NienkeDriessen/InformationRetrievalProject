"""
Module containing functionality to get experiment results.
"""
import json
import os
import pandas as pd

from retrieval import TextToImageRetriever


def execute_experiment(retriever: TextToImageRetriever, query_df: pd.DataFrame, n: int, save_folder: str) -> None:
    """
    For each query, retrieve results and save to json.

    :param retriever: Text-to-image retrieval model.
    :param query_df: Pandas dataframe containing queries.
    :param n: Number of images to retrieve.
    :param save_folder: Folder where to save results.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    res_file_name = os.path.join(save_folder, f"results_{n}.json")

    # Keys are query ids, values are lists of retrieved results
    res = {}
    for _, row in query_df.iterrows():
        query_id = row['id']
        query = row['query']
        results = [retrieval_result.to_dict() for retrieval_result in retriever.retrieve(query, n)]
        res[query_id] = results

    with open(res_file_name, "w") as f:
        json.dump(res, f, indent=4)
