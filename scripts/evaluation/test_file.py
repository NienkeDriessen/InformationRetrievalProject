# import numpy as np
# import pandas as pd
# from pprint import pprint
#
# from scripts.evaluation.evaluation_metrics import evaluate_query, evaluate_all_queries
# from scripts.evaluation.distance_exploration import compute_relative_distances_per_bin, plot_relative_distances
#
# queries = ['query1' , 'query2']
# retrieved_lists = {
#     'query1': {
#         'query': 'query1',
#         'ranked_list': pd.DataFrame([
#             ['image1.jpg', 1],
#             ['image2.jpg', 1],
#             ['image3.jpg', 1],
#             ['image4.jpg', 1],
#             ['image5.jpg', 1],
#             ['image6.jpg', 0],
#             ['image7.jpg', 0],
#             ['image8.jpg', 0],
#             ['image9.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
#     'query2': {
#         'query': 'query2',
#         'ranked_list': pd.DataFrame([
#             ['image1.jpg', 1],
#             ['image2.jpg', 1],
#             ['image11.jpg', 1],
#             ['image3.jpg', 1],
#             ['image4.jpg', 1],
#             ['image12.jpg', 1],
#             ['image5.jpg', 0],
#             ['image6.jpg', 0],
#             ['image7.jpg', 0],
#             ['image13.jpg', 0],
#             ['image8.jpg', 0],
#             ['image9.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
# }
#
# ground_truth = {
#     'query1': {
#         'query': 'query1',
#         'ranked_list': pd.DataFrame([
#             ['image1.jpg', 1],
#             ['image2.jpg', 1],
#             ['image6.jpg', 1],
#             ['image3.jpg', 1],
#             ['image5.jpg', 1],
#             ['image4.jpg', 0],
#             ['image7.jpg', 0],
#             ['image8.jpg', 0],
#             ['image9.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
#     'query2': {
#         'query': 'query2',
#         'ranked_list': pd.DataFrame([
#             ['image9.jpg', 1],
#             ['image2.jpg', 1],
#             ['image3.jpg', 1],
#             ['image1.jpg', 1],
#             ['image5.jpg', 0],
#             ['image6.jpg', 0],
#             ['image7.jpg', 0],
#             ['image8.jpg', 0],
#             ['image4.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
# }
# queries.extend(['query3', 'query4'])
#
# retrieved_lists.update({
#     'query3': {
#         'query': 'query3',
#         'ranked_list': pd.DataFrame([
#             ['image1.jpg', 1],
#             ['image2.jpg', 1],
#             ['image14.jpg', 1],
#             ['image3.jpg', 1],
#             ['image4.jpg', 1],
#             ['image15.jpg', 1],
#             ['image5.jpg', 0],
#             ['image6.jpg', 0],
#             ['image7.jpg', 0],
#             ['image16.jpg', 0],
#             ['image8.jpg', 0],
#             ['image9.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
#     'query4': {
#         'query': 'query4',
#         'ranked_list': pd.DataFrame([
#             ['image1.jpg', 1],
#             ['image2.jpg', 1],
#             ['image17.jpg', 1],
#             ['image3.jpg', 1],
#             ['image4.jpg', 1],
#             ['image18.jpg', 1],
#             ['image5.jpg', 0],
#             ['image6.jpg', 0],
#             ['image7.jpg', 0],
#             ['image19.jpg', 0],
#             ['image8.jpg', 0],
#             ['image9.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
# })
#
# ground_truth.update({
#     'query3': {
#         'query': 'query3',
#         'ranked_list': pd.DataFrame([
#             ['image1.jpg', 1],
#             ['image2.jpg', 1],
#             ['image6.jpg', 1],
#             ['image3.jpg', 1],
#             ['image5.jpg', 1],
#             ['image4.jpg', 0],
#             ['image7.jpg', 0],
#             ['image8.jpg', 0],
#             ['image9.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
#     'query4': {
#         'query': 'query4',
#         'ranked_list': pd.DataFrame([
#             ['image9.jpg', 1],
#             ['image2.jpg', 1],
#             ['image3.jpg', 1],
#             ['image1.jpg', 1],
#             ['image5.jpg', 0],
#             ['image6.jpg', 0],
#             ['image7.jpg', 0],
#             ['image8.jpg', 0],
#             ['image4.jpg', 0],
#             ['image10.jpg', 0],
#         ],
#             columns=['image_path', 'relevance_score'])
#         .set_index('image_path'),
#     },
# })
#
# ks = [1, 3]
# metadata = pd.DataFrame([
#     ['image1.jpg', 'real', ''],
#     ['image2.jpg', 'real', ''],
#     ['image3.jpg', 'real', ''],
#     ['image4.jpg', 'real', ''],
#     ['image5.jpg', 'real', ''],
#     ['image6.jpg', 'real', ''],
#     ['image7.jpg', 'real', ''],
#     ['image8.jpg', 'real', ''],
#     ['image9.jpg', 'real', ''],
#     ['image10.jpg', 'real', ''],
#     ['image11.jpg', 'c1', 'image1.jpg'],
#     ['image12.jpg', 'c1', 'image2.jpg'],
#     ['image13.jpg', 'c2', 'image3.jpg'],
#     ['image14.jpg', 'c2', 'image4.jpg'],
#     ['image15.jpg', 'c3', 'image5.jpg'],
#     ['image16.jpg', 'c3', 'image6.jpg'],
#     ['image17.jpg', 'c2', 'image7.jpg'],
#     ['image18.jpg', 'c3', 'image8.jpg'],
#     ['image19.jpg', 'c2', 'image9.jpg'],
#     ['image20.jpg', 'c1', 'image10.jpg'],
# ],
#     columns=['image_path', 'category', 'og_image'])
#
# scores = evaluate_all_queries(queries, ground_truth, retrieved_lists, ks, metadata)
# pprint(scores)
#
# relative_distances = compute_relative_distances_per_bin(['c1', 'c2', 'c3'], retrieved_lists, metadata)
# print(relative_distances)
#
# plot_relative_distances(relative_distances)
