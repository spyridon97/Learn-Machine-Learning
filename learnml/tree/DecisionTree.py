class DecisionTree:
    """
    :brief: This class is the implementation of the Decision Tree CART algorithm.
    """

    headers = []

    def __init__(self, max_depth=float('inf'), min_samples_split=2):
        """
        :brief: Trivial constructor.

        :param max_depth: is the max depth of the tree. Default infinite.
        :param min_samples_split: is the minimum number of samples required to split an internal node. Default 2.
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class DecisionNode:
        """
        :brief: Decision Node holds a reference to the question, and to
                the two child nodes.
        """

        def __init__(self, question, true_branch, false_branch):
            """
            :brief: Trivial constructor.

            :param question: is the question that this node will ask.
            :param true_branch: is the true branch of this node.
            :param false_branch: is the false branch of this node.
            """
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

    class Leaf:
        """
        :brief: Leaf node holds a dictionary of class -> number of times it
                appears in the rows from the training data that reach this leaf.
                With this, we can classify the data.
        """

        def __init__(self, rows):
            """
            :brief: Trivial constructor.

            :param rows: a list of rows that remained for this leaf.
            """
            self.predictions = DecisionTree.class_counts(rows)

    class Question:
        """
        :brief: The class Question records a column number and a column
                value. The 'match' method is used to compare the feature value
                in an example to the feature value stored in the question.
        """

        def __init__(self, column, value):
            """
            :brief: Trivial constructor.
            :param column: is the given column.
            :param value: is the featured value.
            """
            self.column = column
            self.value = value

        @staticmethod
        def is_numeric(value):
            """
            :brief: This function identifies if a value is numeric or not.

            :param value: is the given value that we will identify.
            :return: a boolean value which indicates if the value was numeric or not.
            """

            return isinstance(value, int) or isinstance(value, float)

        def match(self, example):
            """
            :brief: This function compares the feature value to the feature value
                    in this question.

            :param example: is the given example
            :return: a boolean value which indicates the result of the question
            """

            # Compare the feature value to the feature value in this question.
            val = example[self.column]

            return val >= self.value if self.is_numeric(val) else val == self.value

        def __str__(self):
            """
            :brief: This function helps in print a question in a readable format.

            :return: a printable format for our question.
            """

            condition = ">=" if self.is_numeric(self.value) else "=="

            return "Is %s %s %s?" % (DecisionTree.headers[self.column], condition, str(self.value))

    def __gini(self, rows):
        """
        :brief: This function helps us calculate the Gini Impurity for a list of rows.

        :param rows: is the list of rows
        :return: the gini impurity
        """

        counts = self.class_counts(rows)
        impurity = 1

        for label in counts:
            prob_of_lbl = counts[label] / len(rows)
            impurity -= prob_of_lbl ** 2

        return impurity

    @staticmethod
    def __unique_values(rows, col):
        """
        :brirf: This function detects each unique value for a column in a dataset.

        :param rows: is list of rows.
        :param col: is the given col of which will detect each unique value.
        :return: the unique values of the given column.
        """

        return set([row[col] for row in rows])

    @staticmethod
    def class_counts(rows):
        """
        :brief: This function counts the number of each type of example in a dataset.

        :param rows: is list of rows.
        :return: the counts for each type of example.
        """

        counts = {}  # a dictionary of label -> count.
        for row in rows:
            # The label is always the last column
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1

        return counts

    def __info_gain(self, left, right, current_uncertainty):
        """
        :brief: This function calculates the Information Gain as the uncertainty
                of the starting node, minus the weighted impurity of two child nodes.

        :param left: is the left list of rows.
        :param right: is the right list of rows.
        :param current_uncertainty: is the current uncertainty.
        :return: the info gain.
        """

        p = len(left) / (len(left) + len(right))

        return current_uncertainty - p * self.__gini(left) - (1 - p) * self.__gini(right)

    @staticmethod
    def __divide_set(rows, question):
        """
        :brief: This function divides a dataset by checking for each row
                if it matches the question. If so, we add it to 'true rows',
                otherwise, add it to 'false rows'.

        :param rows: is the dataset that we want to divide.
        :param question: is th question based on which we will divide the set.
        :return: the true and false rows.
        """

        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)

        return true_rows, false_rows

    def __find_best_split(self, rows):
        """
        :brief: This function finds the best question to ask by iterating over
                every feature / value and calculating the information gain.

        :param rows: is a list of rows that we want to split.
        :return: the best_gain value and the best_question question.
        """

        # keep track of the best information gain
        best_gain = 0
        # keep train of the feature / value that produced it
        best_question = None
        current_uncertainty = self.__gini(rows)
        # number of features
        n_features = len(rows[0]) - 1

        for feature in range(n_features):  # for each feature

            values = self.__unique_values(rows, feature)
            for value in values:  # for each value

                question = self.Question(feature, value)

                # we try to split the dataset
                true_rows, false_rows = self.__divide_set(rows, question)

                # we skip this split if it doesn't divide the dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = self.__info_gain(true_rows, false_rows, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def __build_tree(self, rows, depth=1):
        """
        :brief: This function builds the decision tree.

        :param rows: is the given dataset.
        :param depth: is a value which helps us create a tree with specific depth.
        :return: the root of the decision node.
        """

        # First we try partitioning the dataset on each of the unique attribute,
        # Then, we calculate the information gain, and return the question that produces the highest gain.
        gain, question = self.__find_best_split(rows)

        # Base case: no further info gain, Since we can ask no further questions, we'll return a leaf.
        if gain == 0:
            return self.Leaf(rows)

        # If we reach here, we have found a useful feature / value to partition on.
        true_rows, false_rows = self.__divide_set(rows, question)

        # check for max depth
        if depth >= self.max_depth:
            # We make the true rows a leaf.
            true_leaf = self.Leaf(true_rows)

            # We make the false rows a leaf.
            false_leaf = self.Leaf(false_rows)

            # Return a Question node. This records the best feature / value to ask at this point,
            # as well as the branches to follow depending on the answer.
            return self.DecisionNode(question, true_leaf, false_leaf)

        # process left child
        if len(true_rows) <= self.min_samples_split:
            true_branch = self.Leaf(true_rows)
        else:
            true_branch = self.__build_tree(true_rows, depth + 1)

        # process right child
        if len(false_rows) <= self.min_samples_split:
            false_branch = self.Leaf(false_rows)
        else:
            false_branch = self.__build_tree(false_rows, depth + 1)

        # Return a Question node. This records the best feature / value to ask at this point,
        # as well as the branches to follow depending on the answer.
        return self.DecisionNode(question, true_branch, false_branch)

    def fit(self, train_set, headers=None):
        """
        :brief: This function builds the Decision Tree.

        :param: train_set: is the given train set that will be used to train the tree.
        :param: headers: are the columns headers
        """

        if headers is not None:
            DecisionTree.headers = headers
        else:
            DecisionTree.headers = [str(i) for i in range(len(train_set))]

        self.root = self.__build_tree(train_set)

    def __print_tree(self, node, spacing=""):
        """
        :brief: This function helps us print the decision tree that we built.

        :param node:    is the root of the decision tree.
        :param spacing: is the spacing that we want for printing,
        """

        tree_string = ""
        # Base case: we've reached a leaf
        if isinstance(node, self.Leaf):
            tree_string += spacing + "Predict " + str(node.predictions) + '\n'
            return tree_string

        # Print the question at this node
        tree_string += spacing + str(node.question) + '\n'

        # Call this function recursively on the true branch
        tree_string += spacing + '--> True:' + '\n'
        tree_string += self.__print_tree(node.true_branch, spacing + "    ")

        # Call this function recursively on the false branch
        tree_string += spacing + '--> False:' + '\n'
        tree_string += self.__print_tree(node.false_branch, spacing + "    ")
        return tree_string

    def __str__(self):
        """
        :brief: This function prints our Decision Tree Classifier.
        """

        return self.__print_tree(self.root)

    def __classify(self, row, node):
        """
        :brief: This function predicts the result of a row using our decision tree.

        :param row: is the row that we want to predict.
        :param node: is the root of our decision tree.
        :return: the prediction of our decision tree for the given row.
        """

        # Base case: we've reached a leaf
        if isinstance(node, self.Leaf):
            return node.predictions

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node, to the example we're considering.
        if node.question.match(row):
            return self.__classify(row, node.true_branch)
        else:
            return self.__classify(row, node.false_branch)

    def classify(self, row):
        """
        :brief: This function predicts the result of a row using the decision tree.

        :param row: is the row that we want to predict.
        :return: the prediction of our decision tree for the given row.
        """

        return self.__classify(row, self.root)

    def predict(self, test_set):
        """
        :brief: This function predicts the scores of the test set by using the decision tree.

        :param test_set: is the given test that we want to classify.
        :return: the predictions of the decision_tree.
        """

        predictions = []
        for row in test_set:
            if len(self.classify(row).keys()) > 1:
                predictions.append(max(self.classify(row), key=self.classify(row).get))
            else:
                predictions.append(list(self.classify(row).keys())[0])

        return predictions

    def predict_probabilities(self, test_set):
        """
        :brief: This function predicts the probabilities of the test set by using the decision tree.

        :param test_set: is the given test that we want to classify.
        :return: the probabilities of the predictions of the decision tree.
        """

        predictions = [self.classify(row) for row in test_set]
        predictions_probabilities = []

        for prediction in predictions:
            total = sum(prediction.values()) * 1.0
            probabilities = {}
            for label in prediction.keys():
                probabilities[label] = str(int(prediction[label] / total * 100)) + "%"
            predictions_probabilities.append(probabilities)
        return predictions_probabilities
