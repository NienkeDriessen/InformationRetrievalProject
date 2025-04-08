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
        df = results.copy()
        df['rank'] = np.arange(len(df))
        for index, row in df.iloc[:20].iterrows():  # take first 20 ranking results?
            id_to_search_for = os.path.splitext(os.path.split(row.name)[1])[0]  # get the id of the image
            metadata_rows = metadata.loc[metadata['image_id'] == id_to_search_for]
            og_image = metadata_rows['og_image']  # is id
            og_path = metadata_rows['image_path']  # is path

            if len(og_image) == 0:
                print(f'No original image found for this image: {row.name}')
                continue
            og_image = og_image.values[0]  # get the original image path
            if og_image == '':
                print(f'Image should be original: {row.name}')
                continue  # if the image is not altered, skip it as dist = 0 always
            rank = row['rank']

            # get the bin for the image
            bin = metadata.loc[metadata['image_id'] == id_to_search_for, 'ratio_category'].values[0]
            og_rank = df.loc[[df['img_id'] == og_image], 'rank']

            # if the og image is not in the ranked list, skip this image
            if og_rank == 0:  # TODO: OG not in ranking
                print(f'Original image not in ranking: {og_image}')
                continue

            # compute the relative distance
            relative_distance = (rank - og_rank)
            relative_distances[bin].append(relative_distance)
    print(relative_distances)
    return relative_distances


def plot_relative_distances(relative_distances: dict[str, list[float]], save_path: str = '') -> None:
    """
    Plot the relative distances for each bin. Assumes bin names c1, c2, ...
    :param relative_distances: dictionary with bins as keys and relative distances as values
    :param save_path: path to save the plot
    """
    data = [relative_distances[category] for category in ['bin1', 'bin2', 'bin3']]

    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=False, showmedians=True)
    plt.title('Relative distance in rank of altered images compared to their corresponding original image per bin')
    plt.xlabel('Bins')
    plt.ylabel('Relative Distance')
    plt.xticks(range(1, len(relative_distances) + 1), list(relative_distances.keys()))
    plt.grid()
    if save_path != '':
        plt.savefig(os.path.join(save_path, 'relative_distances.png'))
    plt.show()
