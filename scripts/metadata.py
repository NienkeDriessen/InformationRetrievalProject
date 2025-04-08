import pandas as pd


def preprocess_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses metadata.
    For now only removes rows with duplicate image ids.

    :param metadata: Metadata dataframe
    :return: Processed metadata dataframe
    """
    processed_metadata = metadata.drop_duplicates(subset='image_id', keep='first')
    return processed_metadata
