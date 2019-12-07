def accuracy_score(y_true, y_pred):
    """
    :brief: This function calculates the accuracy of a test set.

    :param y_true: are the y values of the dataset
    :param y_pred: are prediction of a classifier.
    :return: the accuracy
    """

    # we check if the predictions were correct or not.
    predictions_check = [i == j for i, j in zip(y_true, y_pred)]

    # The accuracy is not a percentage. In case you want a percentage multiply it by 100.
    accuracy = sum(predictions_check) / len(y_true)

    return accuracy
