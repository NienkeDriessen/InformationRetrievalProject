import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_relative_distances_per_bin(bins: list[str], retrieval_results: dict[str, pd.DataFrame],
                                       metadata: pd.DataFrame) -> dict[str, list[float]]:
    """
    Compute the relative distances for each bin.
    :param bins: list of bins
    :param retrieval_results: retrieval results for each query
    :param metadata: metadata for the images
    :return: dictionary with bins as keys and relative distances as values
    """
    relative_distances = {bin: [] for bin in bins}

    for results in retrieval_results.values():
        df = results['ranked_list'].copy()
        df['rank'] = np.arange(len(df))
        for index, row in df.iterrows():
            og_image = metadata.loc[metadata['image_path'] == row.name, 'og_image'].values[0]  # TODO: error here
            if og_image == '':
                continue  # if the image is not altered, skip it as dist = 0 always
            rank = row['rank']
            # get the bin for the image
            bin = metadata.loc[metadata['image_path'] == row.name, 'category'].values[0]
            og_rank = df.loc[og_image, 'rank']
            # if the og image is not in the ranked list, skip this image
            if og_rank == 0:
                continue
            # compute the relative distance
            relative_distance = (rank - og_rank)
            relative_distances[bin].append(relative_distance)
    return relative_distances


def plot_relative_distances(relative_distances: dict[str, list[float]], save_path: str = '') -> None:
    """
    Plot the relative distances for each bin. Assumes bin names c1, c2, ...
    :param relative_distances: dictionary with bins as keys and relative distances as values
    :param save_path: path to save the plot
    """
    data = [relative_distances[category] for category in ['c1', 'c2', 'c3']]

    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=False, showmedians=True)
    plt.title('Relative distances of altered images to original image per Bin')
    plt.xlabel('Bins')
    plt.ylabel('Relative Distance')
    plt.xticks(range(1, len(relative_distances) + 1), list(relative_distances.keys()))
    plt.grid()
    if save_path != '':
        plt.savefig(os.path.join(save_path, 'relative_distances.png'))
    plt.show()
