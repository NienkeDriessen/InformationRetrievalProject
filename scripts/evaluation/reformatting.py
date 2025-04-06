import pandas as pd


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


def reformat_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Renames and filters the metadata to match the expected format for evaluation.
    :param metadata:
    :return:
    """
    p_image_path = 'image_path'
    p_category = 'ratio_category'
    p_og_image = ''

    return metadata[[p_image_path, p_category, p_og_image]].rename(
        columns={p_image_path: 'image_path', p_category: 'category', p_og_image: 'og_image'}
    ).set_index('image_path')
