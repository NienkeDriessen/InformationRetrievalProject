import pandas as pd


def alt_to_og(query_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the mapping of alternate image id to original id.
    :param query_df: query dataframe
    """
    image_id = []
    og_image = []
    for index, row in query_df.iterrows():
        for i in row['place']:
            image_id.append(row.image_id)
            og_image.append(i)
    return pd.DataFrame({'image_id': image_id, 'og_image': og_image})


def reformat_retrieval_results(retrieval: pd.DataFrame) -> pd.DataFrame:
    """
    Renames and filters the retrieval results to match the expected format for evaluation.
    :param retrieval:
    :return:
    """
    p_img_path = 'img_path'
    p_relevance_score = 'distance'
    return retrieval[[p_img_path, p_relevance_score]].rename(columns={
        p_img_path: 'image_path',
        p_relevance_score: 'relevance_score'
    }).set_index('image_path')


def reformat_metadata(metadata: pd.DataFrame, image_list: list[str],
                      alternate_to_original: pd.DataFrame) -> pd.DataFrame:
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
    mdata = metadata[['image_id', 'image_path', 'ratio_category', 'label', 'entities']].copy()
    mdata = mdata.merge(alternate_to_original, left_on='image_id', right_on='image_id', how='left')
    mdata = mdata.loc[mdata['image_id' in image_list]]
    mask = ((mdata['label'] == 'real') | (mdata['og_image'].notna()))
    mdata = mdata[mask]
    mask = ((mdata['entities'].notna()) & (mdata['label'] == 'real')) | (mdata['label'] == 'fake')
    mdata = mdata[mask]
    return mdata
