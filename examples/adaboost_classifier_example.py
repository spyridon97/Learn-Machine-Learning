import os

from learnml.ensemble import AdaBoostClassifier
from learnml.tree import DecisionTree
from learnml.io import read_dataset
from learnml.metrics import accuracy_score
from learnml.model_selection import train_test_split
from examples.datasets import datasets_path


def main():
    """
    :brief: The main function executes the program.
    """

    filename = os.path.join(datasets_path, 'bank.csv')
    x, y, headers = read_dataset(filename, header=True, x_y=True)

    # we create the train = 70% dataset and the test = 30% dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTree(max_depth=1), n_estimators=5)
    adaboost_classifier.fit(x_train, y_train, headers=headers)
    predictions = adaboost_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy of AdaBoost Classifier: {:.2f}% ".format(accuracy * 100))


if __name__ == '__main__':
    main()