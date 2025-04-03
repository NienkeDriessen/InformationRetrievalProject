"""
Main module, executes the entire experiment.
"""
import ssl

import torch
from clip import clip

from load_data import load_metadata, load_queries_and_image_ids, load_embeddings
from embedding import generate_image_embeddings
from evaluation.ground_truth import gen_ground_truth
import os

from retrieval import TextToImageRetriever
from experiment import execute_experiment
from load_data import save_embeddings

# Defining constants
METADATA_PATH = '../metadata/metadata_OpenImages.csv'
EMBEDDING_FOLDER_PATH = '../data/embeddings'
EMBEDDING_NAME = 'img_embeddings.h5'
EMBEDDING_PATH = os.path.join(EMBEDDING_FOLDER_PATH, EMBEDDING_NAME)
IMG_PATH = '../data_openImages'
QUERY_PATH = '../data/queries_at_least_3_sufficient_altered.csv'
RETRIEVAL_RESULTS_PATH = '../data/retrieval_results'
RESULT_PATH = '../results/'  # path to folder to save metric results and graphs to
K_VALUES = [1, 3, 5, 10]  # values for k used in the metrics in evaluation (EG, nDCG@k)
REGENERATE_EMBEDDINGS = True


def main():
    """
    Main method entrypoint of the application.
    """
    # Loading images and metadata
    metadata = load_metadata(METADATA_PATH)

    # Generating and saving queries
    query_df, image_list = load_queries_and_image_ids(QUERY_PATH)  # columns=[id,keywords,query,num_altered,altered_ids]

    # Define embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssl._create_default_https_context = ssl._create_unverified_context # Disable SSL because CLIP downloading doesn't work sometimes
    model, preprocess = clip.load("ViT-B/32", device=device)
    ssl._create_default_https_context = ssl.create_default_context

    if not os.path.exists(EMBEDDING_FOLDER_PATH) or REGENERATE_EMBEDDINGS:
        if not os.path.exists(EMBEDDING_FOLDER_PATH):
            os.makedirs(EMBEDDING_FOLDER_PATH)
        embeddings, image_paths = generate_image_embeddings(IMG_PATH, metadata, image_list, model, preprocess, device)
        save_embeddings(EMBEDDING_PATH, embeddings, image_paths)
    else:
        embeddings = load_embeddings(EMBEDDING_PATH)

    # Setting up retrieval pipeline
    retriever = TextToImageRetriever(model, device, embeddings)

    # Executing experiment for each K value, saves to json file for each K.
    for k in K_VALUES:
        execute_experiment(retriever, query_df, k, RETRIEVAL_RESULTS_PATH)

    # Evaluating
    ## Generate ground truth
    queries = query_df[['id', 'query']]
    gen_ground_truth(queries, metadata, RESULT_PATH, model, preprocess,  10)

    ## Evaluation script - individual functions detail expected input structure
    # evaluate(K_VALUES, retrieval_results_path: str, metadata, queries, RESULTS_PATH)  # TODO: edit reformat metadata for actual metadata, retrieval results thing

if __name__ == "__main__":
    main()
