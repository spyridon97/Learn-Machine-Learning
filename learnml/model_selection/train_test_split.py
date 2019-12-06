import random


def train_test_split(dataset, test_size=0.3, random_state=None, shuffle=True):
    """
    :brief: This function creates the train and the test sets with
            a specific percentage for the test and the complementary one for
            train set. E.g. test_size = 0.3 means train_set is the 70% of our data
            while test_set is the 30%.

    :param dataset: are the data that we will split.
    :param test_size: is the value based on which we will split the data
    :param random_state: is int number which is used as seed for suffling
    :param shuffle: is a boolean value which indicates if the dataset will be shuffled.
    :return: train_set, test_set
    """

    if random_state is not None:
        random.seed(random_state)

    if shuffle:
        random.shuffle(dataset)

    # we assume that our train_set is 70% of the given dataset and our test_set is the 30% of the given dataset
    train_set = dataset[:int((1 - test_size) * len(dataset))]
    test_set = dataset[int((1 - test_size) * len(dataset)): len(dataset)]

    return train_set, test_set
