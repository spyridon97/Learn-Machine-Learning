class DecisionTree:
    """
    :brief: This class is the implementation of the Decision Tree CART algorithm.
    """

    headers = []

    def __init__(self, max_depth=float('inf'), min_samples_split=2, min_samples_leaf=1):
        """
        :brief: Trivial constructor.

        :param max_depth: is the max depth of the tree. Default infinite.
        :param min_samples_split: is the minimum number of samples required to split an internal node. Default =  2.
        :param min_samples_leaf: is the minimum number of samples required to be at a leaf node. Default = 1.
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
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
                appears in the x from the training data that reach this leaf.
                With this, we can classify the data.
        """

        def __init__(self, y):
            """
            :brief: Trivial constructor.

            :param y: a list of labels that remained for this leaf.
            """
            self.predictions = DecisionTree.class_counts(y)

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

    def __gini(self, y):
        """
        :brief: This function helps us calculate the Gini Impurity for a list of x.

        :param: y are the independent values that will be used to calculate Gini impurity.
        :return: the gini impurity
        """

        counts = self.class_counts(y)
        impurity = 1

        for label in counts:
            probability_of_label = counts[label] / len(y)
            impurity -= probability_of_label ** 2

        return impurity

    @staticmethod
    def __unique_values(x, column):
        """
        :brirf: This function detects each unique value for a column in a dataset.

        :param x: is list of dependent value.
        :param column: is the given col of which will detect each unique value.
        :return: the unique values of the given column.
        """

        return set([row[column] for row in x])

    @staticmethod
    def class_counts(y):
        """
        :brief: This function counts the number of each type of example in a dataset.

        :param y: is list of labels.
        :return: the counts for each type of example.
        """

        counts = {}  # a dictionary of label -> count.
        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def __info_gain(self, y_true, y_false, current_uncertainty):
        """
        :brief: This function calculates the Information Gain as the uncertainty
                of the starting node, minus the weighted impurity of two child nodes.

        :param y_true: is the true list of labels.
        :param y_false: is the false list of labels.
        :param current_uncertainty: is the current uncertainty.
        :return: the info gain.
        """

        p = len(y_true) / (len(y_true) + len(y_false))

        return current_uncertainty - p * self.__gini(y_true) - (1 - p) * self.__gini(y_false)

    @staticmethod
    def __divide_set(x, y, question):
        """
        :brief: This function divides a dataset by checking for each x
                if it matches the question. If so, we add it to 'true x',
                otherwise, add it to 'false x'.

        :param: x: are the dependent values that will be used to divide the set.
        :param: y are the independent values that will be used to divide the set.
        :param question: is th question based on which we will divide the set.
        :return: x_true, y_true, x_false, y_false
        """

        x_true, y_true, x_false, y_false = [], [], [], []
        for row, label in zip(x, y):
            if question.match(row):
                x_true.append(row)
                y_true.append(label)
            else:
                x_false.append(row)
                y_false.append(label)

        return x_true, y_true, x_false, y_false

    def __find_best_split(self, x, y):
        """
        :brief: This function finds the best question to ask by iterating over
                every feature / value and calculating the information gain.

        :param: x: are the dependent values that will be used to find the best split.
        :param: y are the independent values that will be used to find the best split.
        :return: the best_gain value and the best_question question.
        """

        # keep track of the best information gain
        best_gain = 0
        # keep train of the feature / value that produced it
        best_question = None
        current_uncertainty = self.__gini(y)
        # number of features
        n_features = len(x[0])

        for feature_id in range(n_features):
            values = self.__unique_values(x, feature_id)

            for value in values:  # for each value
                question = self.Question(feature_id, value)

                # we try to split the dataset
                _, y_true, _, y_false = self.__divide_set(x, y, question)

                # we skip this split if it doesn't divide the dataset.
                if len(y_true) == 0 or len(y_false) == 0:
                    continue

                # Calculate the information gain from this split
                gain = self.__info_gain(y_true, y_false, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def __build_tree(self, x, y, depth=1):
        """
        :brief: This function builds the decision tree.

        :param: x: are the dependent values that will be used to train the Decision tree.
        :param: y are the independent values that will be used to train the Decision tree.
        :param depth: is a value which helps us create a tree with specific depth.
        :return: the root of the decision node.
        """

        # First we try partitioning the dataset on each of the unique attribute,
        # Then, we calculate the information gain, and return the question that produces the highest gain.
        gain, question = self.__find_best_split(x, y)

        # Base case: no further info gain, Since we can ask no further questions, we'll return a leaf.
        if gain == 0:
            return self.Leaf(y)
        else:
            # If we reach here, we have found a useful feature / value to partition on.
            x_true, y_true, x_false, y_false = self.__divide_set(x, y, question)

            # Check if min sample_leaf restriction is satisfied
            if len(y_true) < self.min_samples_leaf or len(y_false) < self.min_samples_leaf:
                return self.Leaf(y)
            # Check for max depth
            elif depth >= self.max_depth:
                # We make the true x a leaf.
                true_leaf = self.Leaf(y_true)

                # We make the false x a leaf.
                false_leaf = self.Leaf(y_false)

                # Return a Question node. This records the best feature / value to ask at this point,
                # as well as the branches to follow depending on the answer.
                return self.DecisionNode(question, true_leaf, false_leaf)
            else:
                # process left child
                if len(y_true) <= self.min_samples_split:
                    true_branch = self.Leaf(y_true)
                else:
                    true_branch = self.__build_tree(x_true, y_true, depth + 1)

                # process right child
                if len(y_false) <= self.min_samples_split:
                    false_branch = self.Leaf(y_false)
                else:
                    false_branch = self.__build_tree(x_false, y_false, depth + 1)

                # Return a Question node. This records the best feature / value to ask at this point,
                # as well as the branches to follow depending on the answer.
                return self.DecisionNode(question, true_branch, false_branch)

    def fit(self, x_train, y_train, headers=None):
        """
        :brief: This function builds the Decision Tree.

        :param: x_train: are the dependent values that will be used to train the Decision tree.
        :param: y_train are the independent values that will be used to train the Decision tree.
        :param: headers: are the columns headers.
        """

        if headers is not None:
            DecisionTree.headers = headers
        else:
            DecisionTree.headers = [str(i) for i in range(len(x_train))]

        self.root = self.__build_tree(x_train, y_train)

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
        tree_string += spacing + '|--- True:' + '\n'
        tree_string += self.__print_tree(node.true_branch, spacing + "     ")

        # Call this function recursively on the false branch
        tree_string += spacing + '|--- False:' + '\n'
        tree_string += self.__print_tree(node.false_branch, spacing + "     ")
        return tree_string

    def __str__(self):
        """
        :brief: This function is used to display the Decision Tree.
        """

        return self.__print_tree(self.root)

    def __classify(self, x, node):
        """
        :brief: This function predicts the result of a x using our decision tree.

        :param x: is the x that we want to predict.
        :param node: is the root of our decision tree.
        :return: the prediction of our decision tree for the given x.
        """

        # Base case: we've reached a leaf
        if isinstance(node, self.Leaf):
            return node.predictions

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node, to the example we're considering.
        if node.question.match(x):
            return self.__classify(x, node.true_branch)
        else:
            return self.__classify(x, node.false_branch)

    def classify(self, x):
        """
        :brief: This function predicts the result of a x using the decision tree.

        :param x: is the experiment that will be predicted.
        :return: the prediction of the decision tree for the given x.
        """

        return self.__classify(x, self.root)

    def predict(self, x_test):
        """
        :brief: This function predicts the scores of the test set using the decision tree.

        :param x_test: is the given test that we want to classify.
        :return: the predictions of the decision_tree.
        """

        predictions = []
        for x in x_test:
            if len(self.classify(x).keys()) > 1:
                predictions.append(max(self.classify(x), key=self.classify(x).get))
            else:
                predictions.append(list(self.classify(x).keys())[0])

        return predictions

    def predict_probabilities(self, x_test):
        """
        :brief: This function predicts the probabilities of the test set by using the decision tree.

        :param x_test: is the given test set that will be classified.
        :return: the probabilities of the predictions of the decision tree.
        """

        predictions = [self.classify(row) for row in x_test]
        predictions_probabilities = []

        for prediction in predictions:
            total = sum(prediction.values()) * 1.0
            probabilities = {}
            for label in prediction.keys():
                probabilities[label] = str(int(prediction[label] / total * 100)) + "%"
            predictions_probabilities.append(probabilities)
        return predictions_probabilities
