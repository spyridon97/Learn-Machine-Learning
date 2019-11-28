def __column(x, index):
    """
    :brief: Gets a column of a 2D list.

    :param x: is the 2D list
    :param index: is the index of the returned column
    :return: a column of a 2D list
    """

    return [x[i][index] for i in range(len(x))]


def MinMaxScaler(x, min_value=0, max_value=1):
    """
    :brief: Normalizes data using the min max scaling.

    :param x:           are the data
    :param min_value:   is the minimum value
    :param max_value:   is the maximum value
    :return: scaled data
    """

    data_size = len(x)
    dimensions = len(x[0])

    # Compute min value for each dimension
    x_min_values = [min(__column(x, i)) for i in range(dimensions)]
    x_max_values = [max(__column(x, i)) for i in range(dimensions)]

    # Compute scaling factor and scale data
    scales = [0] * dimensions
    for i in range(dimensions):
        scales[i] = (max_value - min_value) / (x_max_values[i] - x_min_values[i])
        for j in range(data_size):
            x[j][i] = scales[i] * x[j][i] + min_value - x_min_values[i] * scales[i]

    return x
