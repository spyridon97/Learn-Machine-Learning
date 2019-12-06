def accuracy_score(y_true, y_pred):
    """
    :brief: This function calculates the accuracy of a test set.

    :param y_true: are the y values of the dataset
    :param y_pred: are prediction of a classifier.
    :return: the accuracy
    """

    # this is the last column of our test set which includes the results of each row
    last_column_values = [row[-1] for row in y_true]

    # we check if the predictions were correct or not.
    predictions_check = [i == j for i, j in zip(last_column_values, y_pred)]

    # The accuracy is not a percentage. In case you want a percentage multiply it by 100.
    accuracy = sum(predictions_check) / len(last_column_values)

    return accuracy
