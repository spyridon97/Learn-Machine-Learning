import pandas


def read_dataset(filename, delimiter=",", header=False):
    """
    :brief: This function reads a file that represents a dataset.

    :param filename: is the name of the file
    :param delimiter: is the delimiter that is used in the file, Default = ","
    :param header: is a boolean value which indicates if the file includes headers. Default = False
    :return: the dataset
    """

    if not header:
        dataset = pandas.read_csv(filename, delimiter=delimiter, header=None).values.tolist()
    else:
        dataset = pandas.read_csv(filename, delimiter=delimiter).values.tolist()
    return dataset
