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
            ind = int(row['img_index'])
            # raise Exception("Debugging")
            og_index = metadata.loc[metadata['index'] == ind, 'og_image']
            if len(og_index) == 0:
                print(f'No original image found for this image: {ind}')
                continue
            og_index = og_index.values[0]  # get the original image path
            # print('og_index: ', og_index)
            # raise Exception("Debugging")

            if np.isnan(og_index):
                print(f'Image should be an original: {ind}')
                continue  # if the image is not altered, skip it as dist = 0 always
            rank = row['rank']
            # get the bin for the image
            # print each column in metadata and its type
            # metadata index = int, og = float
            # df img_index = object -> str?
            # for column in df.columns:
            #     print(f'Column: {column}, Type: {df[column].dtype}')

            bin = metadata.loc[metadata['index'] == ind, 'ratio_category'].values[0]

            # find rank of the original image
            og_rank = df.loc[df['img_index'] == str(int(og_index)), 'rank']
            # print(int(og_index))
            # print(df['img_index' == str(int(og_index))])

            # og_rank = df.loc[df['img_index'] == str(og_index), 'rank']
            # raise Exception("Debugging")

            # if the og image is not in the ranked list, skip this image
            if og_rank.empty:  # TODO: OG not in ranking
                print(f'Original image not in ranking: {og_index}')
                continue

            # compute the relative distance
            relative_distance = (int(rank) - int(og_rank.values[0]))
            relative_distances[bin].append(relative_distance)
    print(relative_distances)
    return relative_distances


def plot_relative_distances(relative_distances: dict[str, list[float]], save_path: str = '') -> None:
    """
    Plot the relative distances for each bin. Assumes bin names c1, c2, ...
    :param relative_distances: dictionary with bins as keys and relative distances as values
    :param save_path: path to save the plot
    """
    data = [relative_distances[category] for category in ['bin1', 'bin2', 'bin3', 'bin4', 'bin5']]

    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=False, showmedians=True)
    plt.title('Relative distance in rank of altered images compared to their corresponding original image per bin')
    plt.xlabel('Ratio Category Bins')
    plt.ylabel('Relative Distance')
    plt.xticks(range(1, len(relative_distances) + 1), list(relative_distances.keys()))
    plt.grid()
    if save_path != '':
        plt.savefig(os.path.join(save_path, 'relative_distances.png'))
    plt.show()
