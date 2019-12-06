import os

from learnml.tree.DecisionTree import DecisionTree
from learnml.io import read_dataset
from learnml.metrics import accuracy_score
from learnml.model_selection import train_test_split
from examples.datasets import datasets_path


def main():
    """
    :brief: The main function executes the program.
    """

    filename = os.path.join(datasets_path, 'heart.csv')
    dataset, labels = read_dataset(filename, header=True)
    # define_column_labels(labels)

    # we create the train = 70% dataset and the test = 30% dataset
    train_set, test_set = train_test_split(dataset, test_size=0.3)

    decision_tree = DecisionTree(max_depth=4)#, min_samples_split=5)
    decision_tree.fit(train_set, labels)
    print(decision_tree)
    predictions = decision_tree.predict(test_set)
    accuracy = accuracy_score(test_set, predictions)
    print("Accuracy of Decision Tree Classifier: {:.2f}% ".format(accuracy * 100))


if __name__ == '__main__':
    main()
