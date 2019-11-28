import csv


def read_dataset(filename, delimiter="\t", headers=False, only_numbers=True):
    """
    :brief: This function reads a file that represents a dataset.

    :param filename: is the name of the file
    :param delimiter: is the delimiter that is used in the file, Default = "\t"
    :param headers: is a boolean value which indicates if the file includes headers. Default = False
    :param only_numbers: is a boolean value which indicates if the file includes only numbers. Default = True
    :return: the dataset
    """

    if not headers:
        reader = csv.reader(open(filename), delimiter=delimiter)
        if only_numbers:
            dataset = [[float(number) for number in row[0].split()] for row in reader]
        else:
            dataset = [[]]
    else:
        if only_numbers:
            dataset = [[]]
        else:
            dataset = [[]]

    return dataset
