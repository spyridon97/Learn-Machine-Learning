import random


def train_test_split(x, y, test_size=0.3, random_state=None, shuffle=True):
    """
    :brief: This function creates the train and the test sets with
            a specific percentage for the test and the complementary one for
            train set. E.g. test_size = 0.3 means x_train is the 70% of our data
            while x_test is the 30%.

    :param x: are the dependent data that will be split.
    :param y: are the independent data that will be split.
    :param test_size: is the value based on which we will split the data
    :param random_state: is int number which is used as seed for suffling
    :param shuffle: is a boolean value which indicates if the dataset will be shuffled.
    :return: x_train, x_test
    """

    if random_state is not None:
        random.seed(random_state)

    if shuffle:
        for i in range(len(y)):
            x[i].append(y[i])
        random.shuffle(x)
        for i in range(len(x)):
            y[i] = x[i][-1]
            x[i] = x[i][:len(x[i]) - 1]

    # we assume that our x_train is 70% of the given dataset and our x_test is the 30% of the given dataset
    x_train = x[:int((1 - test_size) * len(x))]
    y_train = y[:int((1 - test_size) * len(y))]
    x_test = x[int((1 - test_size) * len(x)): len(x)]
    y_test = y[int((1 - test_size) * len(y)): len(y)]

    return x_train, x_test, y_train, y_test
