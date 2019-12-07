from math import sqrt


class AgglomerativeClustering:
    """
    :brief: This class is the implementation of Hierarchical Agglomerative Clustering.
    """

    def __init__(self, n_clusters=2, linkage="single"):
        """
        :brief: This function is the constructor of the Agglomerative Clustering class.

        :param n_clusters: is the number of clusters to form. Default = 2
        :param linkage: is the way of linking the hierarchical agglomerative clusters. Default = "single"
        """

        self.linkage_ways = ["single", "complete", "average", "group average", "centroid"]
        if linkage not in self.linkage_ways:
            print("Linkage way not supported")
            exit(1)

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.clusters_points = []
        self.centers = []
        self.labels = []
        self.SSE = 0

    @staticmethod
    def centroid(points):
        """
        :brief: Computes the centroid of the given points.

        :param points: are the given points
        :return: centroid of the given points
        """

        return [sum([point[i] for point in points]) / len(points) for i in range(len(points[0]))]

    @staticmethod
    def distance(x, y):
        """
        :brief: Finds the Euclidean Distance of 2 Points.

        :param x: is the first point
        :param y: is the second point
        :return: the Euclidean Distance of 2 Points
        """

        return sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))

    def fit(self, x):
        """
        :brief: Produces n Clusters using Hierarchical Agglomerative algorithm.

        :param x: is the dataset that will be used for training
        """

        # Initialize distance matrix, labels, clusters_points_ids and points of each cluster
        self.labels = [i for i in range(len(x))]
        self.clusters_points = [[point] for point in x]
        clusters_points_ids = [[i] for i in range(len(x))]
        distance_matrix = [[self.distance(x[i], x[j]) for j in range(len(x))] for i in range(len(x))]

        while len(self.clusters_points) != self.n_clusters:
            # find min value index
            min_value = float('inf')
            index_i = -1
            index_j = -1
            for i in range(len(distance_matrix)):
                for j in range(len(distance_matrix)):
                    if i != j and distance_matrix[i][j] < min_value:
                        min_value = distance_matrix[i][j]
                        index_i = i
                        index_j = j

            # Note: It is guaranteed that index_i < index_j

            # Combine the points and points ids of combined clusters
            self.clusters_points[index_i] = self.clusters_points[index_i] + self.clusters_points.pop(index_j)
            clusters_points_ids[index_i] = clusters_points_ids[index_i] + clusters_points_ids.pop(index_j)

            # preserve the 2 rows (index_i, index_j) from the distance matrix and pop the index_j row
            rows_i_j = [[distance for distance in distance_matrix[index_i]], distance_matrix.pop(index_j)]

            # Also pop the elements index_j from every other row
            for i in range(len(distance_matrix)):
                del distance_matrix[i][index_j]

            counter = 0
            for i in range(len(rows_i_j[0])):
                if i != index_i and i != index_j:
                    # Single Link / Min
                    if self.linkage == self.linkage_ways[0]:
                        value = min(rows_i_j[0][i], rows_i_j[1][i])
                    # Complete Link / Max
                    elif self.linkage == self.linkage_ways[1]:
                        value = max(rows_i_j[0][i], rows_i_j[1][i])
                    # Average
                    elif self.linkage == self.linkage_ways[2]:
                        value = (rows_i_j[0][i] + rows_i_j[1][i]) / 2
                    # Group Average
                    elif self.linkage == self.linkage_ways[3]:
                        summation = sum([self.distance(point1, point2) for point2 in self.clusters_points[counter]
                                         for point1 in self.clusters_points[index_i]])
                        value = summation / (len(self.clusters_points[counter]) * len(self.clusters_points[index_i]))
                    # Centroid
                    else:
                        value = self.distance(self.centroid(self.clusters_points[counter]),
                                              self.centroid(self.clusters_points[index_i]))

                    distance_matrix[index_i][counter] = value
                    distance_matrix[counter][index_i] = value
                    counter += 1
                elif i == index_i:
                    counter += 1

        # Compute labels
        for i in range(len(clusters_points_ids)):
            for j in range(len(clusters_points_ids[i])):
                self.labels[clusters_points_ids[i][j]] = i

    def fit_predict(self, x):
        """
        :brief: Produces n Clusters using Hierarchical Agglomerative algorithm.

        :param x: is the dataset that will be used for training
        :return: cluster labels of each point
        """

        self.fit(x)

        return self.labels
