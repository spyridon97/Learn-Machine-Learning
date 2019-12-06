import random
from math import sqrt
from collections import defaultdict


class KMeans:
    """
    :brief: This class is the implementation of KMeans.
    """

    def __init__(self, n_clusters=2, random_state=None, max_iter=300):
        """
        :brief: This function is the constructor of the Logistic Regression Classifier class.

        :param n_clusters: is the number of clusters to form as well as the number of centroids to generate. Default = 2
        :param random_state: determines random number generation for centroid initialization.
                             Use an int to make the randomness deterministic
        :param max_iter: Maximum number of iterations of the k-means algorithm for a single run. Default = 3000
        """

        self.n_clusters = n_clusters
        if random_state is not None:
            random.seed(random_state)
        self.max_iter = max_iter
        self.centers = []
        self.labels = []
        self.SSE = 0

    def __generate_k(self, x):
        """
        :brief: Generate k random initial centers from a given data set

        :param x: is the given data set
        """

        self.centers = [random.choice(x) for _ in range(self.n_clusters)]

    @staticmethod
    def centroid(points):
        """
        :brief: Computes the centroid of the given points.

        :param points: are the given points
        :return: centroid of the given points
        """

        return [sum([point[i] for point in points]) / len(points) for i in range(len(points[0]))]

    def __update_centers(self, x, labels):
        """
        :brief: Updates the centers of the clustering. If there are empty clusters, the furthest points from the center
                of the biggest cluster are removed from it and added as new centers of the empty clusters.

        :param x: is the given x
        :param labels: are the current labels of the points to each cluster (index)
        """

        new_means = defaultdict(list)

        for label, point in zip(labels, x):
            new_means[label].append(point)

        # Handling empty clusters
        if self.n_clusters > len(new_means):
            # Calculate cluster sizes
            cluster_sizes = [len(new_means[i]) if i in new_means.keys() else 0 for i in range(self.n_clusters)]

            # Determine biggest cluster id and empty clusters's id
            biggest_cluster_id = cluster_sizes.index(max(cluster_sizes))
            empty_clusters_ids = [i for i in range(len(cluster_sizes)) if cluster_sizes[i] == 0]

            # Find the n points that are the furthest from the biggest cluster center, where n = len(empty_clusters_ids)
            distances = sorted(zip([self.distance(self.centers[biggest_cluster_id], point) for point in
                                    new_means[biggest_cluster_id]], new_means[biggest_cluster_id]))

            furthest_points = [distances[len(distances) - 1 - i][1] for i in range(len(empty_clusters_ids))]

            # Remove those points from the biggest cluster
            for point in furthest_points:
                new_means[biggest_cluster_id].remove(point)

            # Assign the centers and use the extracted points as centers for the empty clusters
            self.centers = []
            counter = 0
            for i in range(self.n_clusters):
                if i not in empty_clusters_ids:
                    self.centers.append(self.centroid(new_means[i]))
                else:
                    self.centers.append(furthest_points[counter])
                    counter += 1
        else:
            self.centers = [self.centroid(new_means[i]) for i in range(self.n_clusters)]

    @staticmethod
    def distance(x, y):
        """
        :brief: Finds the Euclidean Distance of 2 Points.

        :param x: is the first point
        :param y: is the second point
        :return: the Euclidean Distance of 2 Points
        """

        return sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))

    def __assign_points(self, x):
        """
        :brief: Assigns each point of the given data set to a specific center,

        :param x: is the given data set
        :return: an array of indexes of centers that correspond to an index in a point of the data set
        """

        labels = []
        for point in x:
            distances_from_centers = [self.distance(point, self.centers[i]) for i in range(self.n_clusters)]
            cluster_id_with_smallest_distance = distances_from_centers.index(min(distances_from_centers))
            labels.append(cluster_id_with_smallest_distance)

        return labels

    def fit(self, x):
        """
        :brief: Produces K-Clusters using KMeans algorithm.

        :param x: is the dataset that will be used for training
        """

        self.__generate_k(x)

        self.labels = self.__assign_points(x)
        old_labels = None

        iterations = 0
        while self.labels != old_labels and iterations != self.max_iter - 1:
            self.__update_centers(x, self.labels)
            old_labels = self.labels
            self.labels = self.__assign_points(x)
            iterations += 1

        self.SSE = sum([self.distance(x[i], self.centers[self.labels[i]]) for i in range(len(x))])

    def fit_predict(self, x):
        """
        :brief: Produces K-Clusters using KMeans algorithm.

        :param x: is the dataset that will be used for training
        :return: cluster labels of each point
        """

        self.fit(x)

        return self.labels

    def classify(self, x_point):
        """
        :brief: Predicts cluster id of a given point.

        :param x_point: is the point that will predicted
        :return the id of the cluster that the given point is assigned to
        """

        distances_from_centers = [self.distance(x_point, self.centers[i]) for i in range(self.n_clusters)]
        min_distance_id = distances_from_centers.index(min(distances_from_centers))

        return min_distance_id

    def predict(self, points):
        """
        :brief: predicts cluster labels of a given point set

        :param points: are the given points that will be classified in which cluster there are.
        :return: the predictions-cluster labels of the given points
        """

        return [self.classify(point) for point in points]
