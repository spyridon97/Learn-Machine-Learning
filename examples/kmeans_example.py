import os
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, median

from learnml.cluster import KMeans
from learnml.preprocessing import MinMaxScaler
from learnml.io import read_dataset
from examples.datasets import datasets_path


def main():
    """
    :brief: The main function executes the program.
    """

    filename = os.path.join(datasets_path, 'KMeans.csv')
    dataset = read_dataset(filename)
    dataset = MinMaxScaler(dataset, min_value=0, max_value=1)

    # Compute SEE for k in [2, 10]
    mean_SEE_per_k = []
    k_list = [i for i in range(2, 11)]
    for k in k_list:
        # Calculate mean SSE for each k for 80 iterations
        sse_list = []
        for i in range(80):
            cluster = KMeans(n_clusters=k)
            cluster.fit(dataset)
            sse_list.append(cluster.SSE)
        mean_SEE_per_k.append(mean(sse_list))

    # Compute Labels for k = 3 and select the median iteration based on its SSE. Iterations = 50
    sse_list = []
    labels = []
    for i in range(50):
        cluster = KMeans(n_clusters=3)
        labels.append(cluster.fit_predict(dataset))
        sse_list.append(cluster.SSE)

    id_with_median_sse = sse_list.index(median(sse_list))
    labels = labels[id_with_median_sse]

    # Plot everything
    fig, subplots = plt.subplots(nrows=1, ncols=2)

    # Plot SSE of KMeans
    subplots[0].plot(k_list, mean_SEE_per_k)
    subplots[0].set_title('KMeans - SSE per Number of clusters(k)')

    # Plot labels of KMeans
    dataset = np.array(dataset)
    subplots[1].scatter(dataset[:, 0], dataset[:, 1], c=labels)
    subplots[1].set_title('KMeans - Clustering Labels')
    plt.show()


if __name__ == '__main__':
    main()
