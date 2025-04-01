import numpy as np
import pandas as pd
import pprint

from scripts.evaluation.evaluation import evaluate_query, evaluate_all_queries

queries = ['query1', 'query2']
ground_truth = {
    'query1': {
        'query': 'query1',
        'ranked_list': pd.DataFrame([
            ['image1.jpg', 1],
            ['image2.jpg', 1],
            ['image3.jpg', 1],
            ['image4.jpg', 1],
            ['image5.jpg', 1],
            ['image6.jpg', 0],
            ['image7.jpg', 0],
            ['image8.jpg', 0],
            ['image9.jpg', 0],
            ['image10.jpg', 0],
        ],
            columns=['image_path', 'relevance_score'])
        .set_index('image_path'),
    },
    'query2': {
        'query': 'query2',
        'ranked_list': pd.DataFrame([
            ['image1.jpg', 1],
            ['image2.jpg', 1],
            ['image3.jpg', 1],
            ['image4.jpg', 1],
            ['image5.jpg', 0],
            ['image6.jpg', 0],
            ['image7.jpg', 0],
            ['image8.jpg', 0],
            ['image9.jpg', 0],
            ['image10.jpg', 0],
        ],
            columns=['image_path', 'relevance_score'])
        .set_index('image_path'),
    },
}

retrieved_lists = {
    'query1': {
        'query': 'query1',
        'ranked_list': pd.DataFrame([
            ['image1.jpg', 1],
            ['image2.jpg', 1],
            ['image6.jpg', 1],
            ['image3.jpg', 1],
            ['image5.jpg', 1],
            ['image4.jpg', 0],
            ['image7.jpg', 0],
            ['image8.jpg', 0],
            ['image9.jpg', 0],
            ['image10.jpg', 0],
        ],
            columns=['image_path', 'relevance_score'])
        .set_index('image_path'),
    },
    'query2': {
        'query': 'query2',
        'ranked_list': pd.DataFrame([
            ['image9.jpg', 1],
            ['image2.jpg', 1],
            ['image3.jpg', 1],
            ['image1.jpg', 1],
            ['image5.jpg', 0],
            ['image6.jpg', 0],
            ['image7.jpg', 0],
            ['image8.jpg', 0],
            ['image4.jpg', 0],
            ['image10.jpg', 0],
        ],
            columns=['image_path', 'relevance_score'])
        .set_index('image_path'),
    },
}

ks = [1, 3]
metadata = pd.DataFrame([
    ['image1.jpg', 'real', ''],
    ['image2.jpg', 'real', ''],
    ['image3.jpg', 'real', ''],
    ['image4.jpg', 'real', ''],
    ['image5.jpg', 'real', ''],
    ['image6.jpg', 'real', ''],
    ['image7.jpg', 'real', ''],
    ['image8.jpg', 'real', ''],
    ['image9.jpg', 'real', ''],
    ['image10.jpg', 'real', ''],
],
    columns=['image_path', 'category', 'og_image'])

scores = evaluate_all_queries(queries, ground_truth, retrieved_lists, ks, metadata)
pprint(scores)

