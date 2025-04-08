import pandas as pd
import numpy as np


def alt_to_og(query_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the mapping of alternate image index to original index.
    :param query_df: query dataframe
    """
    image_id = []
    og_image = []
    for index, row in query_df.iterrows():
        for i in row['altered_indices']:
            image_id.append(np.float64(row['index']))
            og_image.append(np.float64(i))
    return pd.DataFrame({'index': image_id, 'og_image': og_image})


def reformat_retrieval_results(retrieval: pd.DataFrame) -> pd.DataFrame:
    """
    Renames and filters the retrieval results to match the expected format for evaluation.
    :param retrieval:
    :return:
    """
    return retrieval.rename(columns={
        'image_index': 'index',
        'distance': 'relevance_score'
    })


def reformat_metadata(metadata: pd.DataFrame, image_list: list[str]) -> pd.DataFrame:
    """
    Reformats OpenImage's metadata to the following:
    - removes unnecessary columns for ground truth and evaluation
    - merges the original images into the metadata as the 'og_image' column
    - filter out images not associated to queries
    - filter out altered images with no original images
    - filter out real images with no entities
    :param metadata: original metadata dataframe loaded from csv
    :param image_list: list of image ids to keep
    :param alternate_to_original: dataframe with the mapping of alternate image id to original id
    :return: new metadata dataframe with the following columns: ['image_id', 'image_path', 'ratio_category', 'label', 'entities']
    """
    mdata = metadata[['index', 'image_id', 'image_path', 'ratio_category', 'label', 'entities', 'og_image']].copy()

    mask = ((mdata['label'] == 'real') | (mdata['og_image'].notna()))
    mdata = mdata[mask]
    print(len(mdata))

    mask = ((mdata['entities'].notna()) & (mdata['label'] == 'real')) | (mdata['label'] == 'fake')
    mdata = mdata[mask]
    print(len(mdata))

    mask = (mdata['image_id'].isin(image_list) | mdata['og_image'].isin(image_list))
    mdata = mdata.loc[mask]  # ids baaaaad >:(

    mdata.to_csv('metadata_reformatted_HELP.csv', index=False)
    return mdata
