import pandas as pd
import pprint
import json

from scripts.evaluation.evaluation_metrics import evaluate_all_queries
from scripts.evaluation.ground_truth import gen_ground_truth
from scripts.evaluation.reformatting import reformat_retrieval_results, reformat_metadata
from scripts.evaluation.distance_exploration import compute_relative_distances_per_bin, plot_relative_distances

"""
Main script for evaluating the retrieval results of the image retrieval system.

Expected format of retrieval results: JSON file with the following structure:
{ <image_path_from_query>: pd.DataFrame[image_path, relevance_score] }
"""

k_values = [1, 3, 5, 10]  # k values for nDCG@k

RETRIEVAL_RESULTS_PATH = ''  # Fill in
METADATA_PATH = ''
QUERIES_PATH = ''
SAVE_FOLDER = ''

queries = pd.read_csv(QUERIES_PATH)
metadata = pd.read_csv(METADATA_PATH)
reformat_metadata(metadata)
ground_truth = gen_ground_truth(queries, metadata=metadata, ps_rel=10, save_path=SAVE_FOLDER)
print(f'Saved generated ground truth to {SAVE_FOLDER} as a JSON file.')
print('Structure of path:\n <image_path>: { query: str, ranked_list: pd.DataFrame[image_path, relevance_score] }')

# retrieval_results = pd.read_csv(RETRIEVAL_RESULTS_PATH)  # TODO: is this per query or all queries?
retrieval_results = json.load(open(RETRIEVAL_RESULTS_PATH, 'r'))
retrieval_results = reformat_retrieval_results(retrieval_results)

evaluation_results = evaluate_all_queries(queries['image_path'].tolist(), ground_truth, retrieval_results, k_values,
                                          metadata=metadata, save_folder=SAVE_FOLDER)
print(f'Saved generated ground truth to {SAVE_FOLDER} as a JSON file.')
print('Structure of path:\n { <k>: { ndcg_[all|c1|c2|c3|og]: float, relD_[og|c1|c2|c3]_[c1|c2|c3]: float } }')
pprint(evaluation_results)

relative_distances = compute_relative_distances_per_bin(['c1', 'c2', 'c3'], retrieval_results, metadata)
print('Relative distances per bin:')
pprint(relative_distances)

plot_relative_distances(relative_distances, save_path=SAVE_FOLDER)

