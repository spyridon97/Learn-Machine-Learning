import pandas


def read_dataset(filename, delimiter=",", header=False, x_y=False):
    """
    :brief: This function reads a file that represents a dataset.

    :param filename: is the name of the file
    :param delimiter: is the delimiter that is used in the file. Default = ","
    :param header: is a boolean value which indicates if the file includes headers. Default = False
    :param x_y: is a boolean value that indicates if the dataset will be split to x and y. Default = True
    :return: dataset or x, y, and headers if asked
    """

    if not header:
        dataset = pandas.read_csv(filename, delimiter=delimiter, header=None)#.values.tolist
        if x_y:
            x = dataset.iloc[:, :-1].values.tolist()
            y = dataset.iloc[:, 1].values.tolist()
            return x, y
        else:
            dataset = dataset.values.tolist()
            return dataset
    else:
        dataset = pandas.read_csv(filename, delimiter=delimiter)
        headers = dataset.columns.values.tolist()
        if x_y:
            x = dataset.iloc[:, :-1].values.tolist()
            y = dataset.iloc[:, -1].values.tolist()
            return x, y, headers
        else:
            dataset = dataset.values.tolist()
            return dataset, headers
