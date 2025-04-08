"""
Main module, executes the entire experiment.
"""
import h5py
# import ssl

import torch
from clip import clip
import json
import pandas as pd

from load_data import load_metadata, load_queries_and_image_indices, load_embeddings, save_embeddings, save_metadata
from embedding import generate_image_embeddings
from evaluation.ground_truth import gen_ground_truth, gen_entity_embeddings
from evaluation.evaluation import evaluate
from evaluation.reformatting import reformat_metadata
import os

from retrieval import TextToImageRetriever
from experiment import execute_experiment
from load_data import save_embeddings
from evaluation.reformatting import alt_to_og
# from metadata import preprocess_metadata

# from final_query_selection_and_plot_generation import altered_to_og

# Defining constants
METADATA_PATH = '../metadata/metadata_OpenImages.csv'
PROCESSED_METADATA_PATH = '../metadata/processed_metadata_OpenImages.csv'
EMBEDDING_FOLDER_PATH = '../data/embeddings'
EMBEDDING_NAME = 'img_embeddings.h5'
EMBEDDING_PATH = os.path.join(EMBEDDING_FOLDER_PATH, EMBEDDING_NAME)
IMG_PATH = '../data_openImages'
# QUERY_PATH = '../data/queries_at_least_3_sufficient_altered_ratio.csv'
QUERY_PATH = '../data/query_information.csv'
QUERY_EMBEDDINGS = '../data/query_embeddings.h5'
METADATA_W_OG_PATH = '../data/metadata_w_og_images.csv'
RETRIEVAL_RESULTS_PATH = '../data/retrieval_results'
RESULT_PATH = '../results/'  # path to folder to save metric results and graphs to
K_VALUES = [1, 3, 5, 10]  # values for k used in the metrics in evaluation (EG, nDCG@k)
REGENERATE_EMBEDDINGS = True
REGENERATE_RESULTS = True


def main():
    """
    Main method entrypoint of the application.
    """
    # Loading original images and metadata
    metadata = load_metadata(METADATA_PATH)
    # Save metadata to new file (including index serving as id) and load it again
    #metadata = preprocess_metadata(metadata)
    # save_metadata(metadata, PROCESSED_METADATA_PATH)
    # metadata = load_metadata(PROCESSED_METADATA_PATH)
    metadata = pd.read_csv(METADATA_W_OG_PATH)

    # Generating and saving queries
    # query_df, image_list = load_queries_and_image_indices(QUERY_PATH, metadata)
    # columns=[id,keywords,query,num_altered,altered_ids]
    query_df = pd.read_csv(QUERY_PATH)
    query_df['altered_indices'] = query_df['altered_indices'].apply(
        lambda x : list(map(int, x.replace("'", "").replace("[", "").replace("]", "").split(", "))))
    image_list = query_df['index'].tolist()
    [image_list.extend(l) for l in query_df['altered_indices'].tolist()]

    mask = (metadata['index'].isin(image_list)) | (metadata['og_image'].isin(image_list))

    metadata = metadata[mask]
    # print(metadata)
    # raise Exception("Debugging")
    image_id_list = [str(metadata[metadata['index'] == index]['image_id'].iloc[0]) for index in image_list]
    image_ids_to_real_indices = {}
    for i in range(len(image_id_list)):
        image_ids_to_real_indices[image_id_list[i]] = image_list[i]

    # Define embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ssl._create_default_https_context = ssl._create_unverified_context
    # Disable SSL because CLIP downloading doesn't work sometimes
    model, preprocess = clip.load("ViT-B/32", device=device)
    # ssl._create_default_https_context = ssl.create_default_context

    # mdata = metadata[['image_path', 'image_id', 'ratio_category', 'label', 'entities']].copy()
    # mdata['og_image'] = ''
    # altered_to_og = pd.read_csv(os.path.join('../data_plots/', 'altered_to_og.csv'), index_col=0)
    # mdata.merge(altered_to_og, left_on='image_id', right_index=True, how='left')
    # print(mdata.columns)
    # print(mdata[:10])

    if not os.path.exists(EMBEDDING_PATH) or REGENERATE_EMBEDDINGS:
        print(f'Generating embeddings for {len(image_list)} images...')
        if not os.path.exists(EMBEDDING_FOLDER_PATH):
            os.makedirs(EMBEDDING_FOLDER_PATH)
        embeddings, image_indices = generate_image_embeddings(IMG_PATH, metadata, image_ids_to_real_indices, model, preprocess, device)
        save_embeddings(EMBEDDING_PATH, embeddings, image_indices)
        #mdata.to_csv(os.path.join(RESULT_PATH, 'temp_metadata.csv'), index=True)

    # metadata = pd.read_csv(os.path.join(RESULT_PATH, 'tempie6_may_be_good.csv'))
    embeddings = load_embeddings(EMBEDDING_PATH)
    print("embeddings: ", embeddings)
    key = [key for key in embeddings.keys()][0]
    print("first key: ", str(key))
    print("first embedding: ", embeddings[key])
    metadata = pd.read_csv(os.path.join(RESULT_PATH, 'processed_metadata_OpenImages.csv'))
    # TODO: smth with query_generation methods

    if not os.path.exists(RETRIEVAL_RESULTS_PATH):
        os.makedirs(RETRIEVAL_RESULTS_PATH)
    if len(os.listdir(RETRIEVAL_RESULTS_PATH)) == 0 or REGENERATE_RESULTS:  # check the retrieval_results folder is empty
        # Setting up retrieval pipeline
        retriever = TextToImageRetriever(model, device, embeddings)
        # Executing experiment for each K value, saves to json file for each K.
        for k in [len(embeddings)]:  # K_VALUES:
            execute_experiment(retriever, query_df, k, RETRIEVAL_RESULTS_PATH)

    embeddings = None  # clear memory because RIP my RAM

    # ---------- Evaluating ----------
    # ato = alt_to_og(query_df)
    metadata = pd.read_csv(METADATA_W_OG_PATH)
    mdata = metadata  # reformat_metadata(metadata, image_list)
    # raise Exception("Debugging")

    # Generate ground truth
    queries = query_df
    if not os.path.exists(os.path.join(RESULT_PATH, 'queries_w_ground_truth.json')):
        if not os.path.exists(QUERY_EMBEDDINGS):
            entity_embeddings = gen_entity_embeddings(mdata, EMBEDDING_FOLDER_PATH, model, preprocess)
        else:
            entity_embeddings = load_embeddings(QUERY_EMBEDDINGS)
        ground_truth = gen_ground_truth(mdata, queries, entity_embeddings, model, preprocess, RESULT_PATH)
    else:
        with open(os.path.join(RESULT_PATH, 'queries_w_ground_truth.json'), 'r') as f:
            ground_truth = json.load(f)
            ground_truth = {k: pd.DataFrame(v) for k, v in ground_truth.items()}

    # raise Exception("debugging")

    # Evaluation script - individual functions detail expected input structure
    for f in os.listdir(RETRIEVAL_RESULTS_PATH):
        if f.endswith('.json'):
            retrieval_results_path = os.path.join(RETRIEVAL_RESULTS_PATH, f)
            # Evaluate the retrieval results
            evaluate(K_VALUES, retrieval_results_path, mdata, queries, ground_truth, RESULT_PATH)


if __name__ == "__main__":
    main()
