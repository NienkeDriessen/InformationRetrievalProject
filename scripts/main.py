import torch
from clip import clip

from load_data import load_metadata, load_images_and_generate_embeddings, load_queries_and_image_ids, \
    load_embeddings
from evaluation.ground_truth import gen_ground_truth
import os

from retrieval import TextToImageRetriever

METADATA_PATH = '../metadata/metadata_OpenImages.csv'
PATH_TO_EMBEDDINGS = '../data/embeddings/img_embeddings.h5'
IMG_PATH = 'TODO DEFINE THIS' # TODO define this Mohit
QUERY_PATH = '../data/queries_at_least_3_sufficient_altered.csv'
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
    # TODO: Nienke will write script for query loading save as json/dictionary
    query_df, image_list = load_queries_and_image_ids(QUERY_PATH)  # columns=[id,keywords,query,num_altered,altered_ids]

    # Define embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if not os.path.exists(PATH_TO_EMBEDDINGS) or REGENERATE_EMBEDDINGS:
        if not os.path.exists(PATH_TO_EMBEDDINGS):
            os.makedirs(PATH_TO_EMBEDDINGS)
        embeddings = load_images_and_generate_embeddings(IMG_PATH, metadata, PATH_TO_EMBEDDINGS, model)  #TODO -> h5py datastructure
    else:
        embeddings = load_embeddings(PATH_TO_EMBEDDINGS)

    # Setting up retrieval pipeline
    retriever = TextToImageRetriever(model, device, embeddings)

    # Evaluating
    ## Generate ground truth
    queries = query_df[['id', 'query']]
    gen_ground_truth(queries, metadata, RESULT_PATH, model, preprocess,  10)

    ## Evaluation script - individual functions detail expected input structure
    # evaluate(K_VALUES, retrieval_results_path: str, metadata, queries, RESULTS_PATH)  # TODO: edit reformat metadata for actual metadata, retrieval results thing

if __name__ == "__main__":
    main()
