import pandas as pd


def reformat_retrieval_results(retrieval: pd.DataFrame) -> pd.DataFrame:
    """
    Renames and filters the retrieval results to match the expected format for evaluation.
    :param retrieval:
    :return:
    """
    p_img_path = ''
    p_relevance_score = ''
    return retrieval[[p_img_path, p_relevance_score]].rename(columns={
        p_img_path: 'image_path',
        p_relevance_score: 'relevance_score'
    }).set_index('image_path')
