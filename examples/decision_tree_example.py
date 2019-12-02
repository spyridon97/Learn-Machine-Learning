import os

from algorithms.tree.DecisionTree import DecisionTree
from utils.io import read_dataset
from utils.metrics import accuracy_score
from utils.model_selection import train_test_split
from datasets import datasets_path


def main():
    """
    :brief: The main function executes the program.
    """

    filename = os.path.join(datasets_path, 'heart.csv')
    dataset, labels = read_dataset(filename, header=True)
    #define_column_labels(labels)

    # we create the train = 70% dataset and the test = 30% dataset
    print(len(dataset))
    train_set, test_set = train_test_split(dataset, test_size=0.3)#, shuffle=False)
    print(len(train_set) + len(test_set))

    decision_tree = DecisionTree(max_depth=4, min_size=5)
    decision_tree.fit(train_set, labels)
    print(decision_tree)
    predictions = decision_tree.predict(test_set)
    accuracy = accuracy_score(test_set, predictions)
    print("Accuracy of Decision Tree Classifier: {:.2f}% ".format(accuracy * 100))


if __name__ == '__main__':
    main()
