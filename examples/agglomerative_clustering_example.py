import os
import matplotlib.pyplot as plt
import numpy as np

from algorithms.cluster import AgglomerativeClustering
from utils.preprocessing import MinMaxScaler
from utils.io import read_dataset
from examples.datasets import datasets_path


def main():
    """
    :brief: The main function executes the program.
    """

    filename = os.path.join(datasets_path, 'AgglomerativeClustering.txt')
    dataset = read_dataset(filename)
    dataset = MinMaxScaler(dataset, min_value=0, max_value=1)

    labels_hierarchical = []
    linkage_ways = ["single", "complete", "average", "group average", "centroid"]
    for linkage_way in linkage_ways:
        hierarchical_clustering = AgglomerativeClustering(n_clusters=2, linkage=linkage_way)
        labels_hierarchical.append(hierarchical_clustering.fit_predict(dataset))

    # Plot everything
    fig, subplots = plt.subplots(nrows=1, ncols=5)

    # Plot Hierarchical Agglomeration Clustering
    dataset_hierarchical = np.array(dataset)
    for i in range(len(linkage_ways)):
        subplots[i].set_title("HAC: " + linkage_ways[i])
        subplots[i].scatter(dataset_hierarchical[:, 0], dataset_hierarchical[:, 1], c=labels_hierarchical[i])
    plt.show()


if __name__ == '__main__':
    main()
