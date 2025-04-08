"""
Main module, executes the entire experiment.
"""
# import ssl

import torch
from clip import clip
import json
import pandas as pd

from load_data import load_metadata, load_queries_and_image_ids, load_embeddings, save_embeddings, save_metadata
from embedding import generate_image_embeddings
from evaluation.ground_truth import gen_ground_truth
from evaluation.evaluation import evaluate
from evaluation.reformatting import reformat_retrieval_results, reformat_metadata
import os

from retrieval import TextToImageRetriever
from experiment import execute_experiment

# Defining constants
METADATA_PATH = '../metadata/metadata_OpenImages.csv'
PROCESSED_METADATA_PATH = '../metadata/processed_metadata_OpenImages.csv'
EMBEDDING_FOLDER_PATH = '../data/embeddings'
EMBEDDING_NAME = 'img_embeddings.h5'
EMBEDDING_PATH = os.path.join(EMBEDDING_FOLDER_PATH, EMBEDDING_NAME)
IMG_PATH = '../data_openImages'
QUERY_PATH = '../data/queries_at_least_3_sufficient_altered.csv'
RETRIEVAL_RESULTS_PATH = '../data/retrieval_results'
RESULT_PATH = '../results/'  # path to folder to save metric results and graphs to
K_VALUES = [1, 3, 5, 10]  # values for k used in the metrics in evaluation (EG, nDCG@k)
REGENERATE_EMBEDDINGS = False


def main():
    """
    Main method entrypoint of the application.
    """
    # Loading original images and metadata
    metadata = load_metadata(METADATA_PATH)
    # Save metadata to new file (including index serving as id) and load it again
    save_metadata(metadata, PROCESSED_METADATA_PATH)
    metadata = load_metadata(PROCESSED_METADATA_PATH)

    # Generating and saving queries
    query_df, image_list = load_queries_and_image_ids(QUERY_PATH)  # columns=[id,keywords,query,num_altered,altered_ids]

    # Define embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ssl._create_default_https_context = ssl._create_unverified_context # Disable SSL because CLIP downloading doesn't work sometimes
    model, preprocess = clip.load("ViT-B/32", device=device)
    # ssl._create_default_https_context = ssl.create_default_context

    if not os.path.exists(EMBEDDING_FOLDER_PATH) or REGENERATE_EMBEDDINGS:
        if not os.path.exists(EMBEDDING_FOLDER_PATH):
            os.makedirs(EMBEDDING_FOLDER_PATH)
        embeddings, image_indices = generate_image_embeddings(IMG_PATH, metadata, image_list, model, preprocess, device)
        save_embeddings(EMBEDDING_PATH, embeddings, image_indices)
        #mdata.to_csv(os.path.join(RESULT_PATH, 'temp_metadata.csv'), index=True)

    embeddings = load_embeddings(EMBEDDING_PATH)
    #metadata = pd.read_csv(os.path.join(RESULT_PATH, 'temp_metadata.csv'))

    if not os.path.exists(RETRIEVAL_RESULTS_PATH):
        os.makedirs(RETRIEVAL_RESULTS_PATH)
    if len(os.listdir(RETRIEVAL_RESULTS_PATH)) == 0:  # check the retrieval_results folder is empty
        # Setting up retrieval pipeline
        retriever = TextToImageRetriever(model, device, embeddings)
        # Executing experiment for each K value, saves to json file for each K.
        for k in K_VALUES:
            execute_experiment(retriever, query_df, k, RETRIEVAL_RESULTS_PATH)

    # ----- Evaluating -----
    # Generate ground truth
    queries = query_df[['id', 'query']]
    metadata = reformat_metadata(metadata)
    if not os.path.exists(os.path.join(RESULT_PATH, 'queries_w_ground_truth.json')):
        ground_truth = gen_ground_truth(queries, metadata, RESULT_PATH, model, preprocess,  10)
    else:
        with open(os.path.join(RESULT_PATH, 'queries_w_ground_truth.json'), 'r') as f:
            ground_truth = json.load(f)

    # Evaluation script - individual functions detail expected input structure
    for f in os.listdir(RETRIEVAL_RESULTS_PATH):
        if f.endswith('.json'):
            retrieval_results_path = os.path.join(RETRIEVAL_RESULTS_PATH, f)
            # Evaluate the retrieval results
            evaluate(K_VALUES, retrieval_results_path, metadata, queries, ground_truth, RESULT_PATH)


if __name__ == "__main__":
    main()
