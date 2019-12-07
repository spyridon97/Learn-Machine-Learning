from learnml.tree import DecisionTree
from learnml.metrics import accuracy_score
from math import log, exp
import random


class AdaBoostClassifier:
    """
    :brief: This class is the implementation of AdaBoost classifier.
            that uses a number of weak classifiers in ensemble to make
            a strong classifier. We use  decision stumps, which are 1-level
            Decision Trees.
    """

    def __init__(self, base_estimator=None, n_estimators=50, random_state=None):
        """
        :brief: This function is the constructor of the AdaBoost classifier.

        :param base_estimator: is the classifier that will be used for the AdaBoost ensemble model. Default =
        :param n_estimators: is the number of models that will be created.
        :param random_state: determines random number generation for centroid initialization.
                            Use an int to make the randomness deterministic
        """

        if base_estimator is None:
            self.base_estimator = DecisionTree(max_depth=1)
        else:
            self.base_estimator = base_estimator

        # the number of models that we will create.
        self.n_estimators = n_estimators

        if random_state is not None:
            random.seed(random_state)

        # a list of the estimators that we will build
        self.estimators = []

        # a list of the amount of says for each classifier
        self.amount_of_say_values = []

    @staticmethod
    def __check_dataset_suitable_for_adaboost(y):
        """
        :brief: This function checks if the data set is suitable for AdaBoost
                which means that the labels should only have up to 2 unique values.

        :param y: are the labels of the dataset
        :return: a value which indicates if the data set is suitable for AdaBoost
        """

        return 1 <= len(set(y)) <= 2

    @staticmethod
    def __create_train_set_based_on_weights(x_train, y_train, weights):
        """
        :brief: This functions creates a new train set base on sampling with
                repositioning from the training set, with the possibility of
                choosing each example to be proportional to its weight, so that
                a new set of training with the same number of examples arises.

        :param: x_train: are the dependent values that will be used to train the AdaBoost Classifier.
        :param: y_train are the independent values that will be used to train the AdaBoost Classifier.
        :param weights:     is a list of weights which we will use to build our new train_set.
        :return: the new_train_set.
        """

        sum_of_weights = 0
        borders_of_weights = []
        new_x_train = []
        new_y_train = []

        for i in range(len(weights)):
            sum_of_weights += weights[i]
            borders_of_weights.append(sum_of_weights)

        # we create the new_x_train and new_y_train with random distribution
        random_numbers = []
        for i in range(len(x_train)):
            random_number = random.random()
            random_numbers.append(random_number)
            for j in range(len(borders_of_weights)):
                if random_number <= borders_of_weights[j]:
                    new_x_train.append(x_train[j])
                    new_y_train.append(y_train[j])
                    break

        return new_x_train, new_y_train

    def fit(self, x_train, y_train, headers=None):
        """
        :brief: This function builds the AdaBoost N models.

        :param: x_train: are the dependent values that will be used to train the AdaBoost Classifier.
        :param: y_train are the independent values that will be used to train the AdaBoost Classifier.
        :param: headers: are the columns headers.
        """

        if not self.__check_dataset_suitable_for_adaboost(y_train):
            print("Dataset has more than 2 unique labels. Use another dataset.")
            exit(1)

        for i in range(self.n_estimators):
            estimator = self.base_estimator
            estimator.fit(x_train, y_train, headers=headers)
            self.estimators.append(estimator)

            # we find the predictions of the classifier
            predictions = self.estimators[i].predict(x_train)

            # we find the total error and the predictions check of the classifier
            predictions_check = [i == j for i, j in zip(y_train, predictions)]
            total_error = 1 - accuracy_score(y_train, predictions)

            # if total_error > 0.5 or == 0 then we do not build more models using this train_Set
            if total_error > 0.5 or total_error == 0:
                print("The maximum number of models that we should build is {0}.\r\n".format(i))
                self.estimators.pop()
                break

            # we calculate the amount of say for the current model
            self.amount_of_say_values.append(0.5 * log((1.0 - total_error) / (total_error + 1e-10)))

            # there is no need to build the last train_Set
            if i < self.n_estimators - 1:
                # we calculate the weights for each row of our train set
                weights = [1.0 / len(x_train)] * len(x_train)
                for j in range(len(predictions_check)):
                    if predictions_check[j]:
                        weights[j] *= exp(-self.amount_of_say_values[i])
                    else:
                        weights[j] *= exp(self.amount_of_say_values[i])

                # we normalize the results
                weights = [weight / sum(weights) for weight in weights]

                # we create the new train_set
                x_train, y_train = self.__create_train_set_based_on_weights(x_train, y_train, weights)

    def classify(self, x):
        """
        :brief: This function classifies the result of a row using our AdaBoostM1 model.

        :param x: is the row that we want to predict.
        :return: the prediction of our AdaBoostM1 model for the given row.
        """

        # we calculate the predictions of each stump.
        predictions = [decision_stump.classify(x) for decision_stump in self.estimators]

        # we create of set of from our predictions.
        unique_predictions = list(set(predictions))

        # we calculate the amount_of_say pre unique prediction.
        amount_of_say_per_unique_prediction = []
        for i in range(len(unique_predictions)):
            amount_of_say_per_unique_prediction.append(0)

            for j in range(len(predictions)):
                if predictions[j] == unique_predictions[i]:
                    amount_of_say_per_unique_prediction[i] += self.amount_of_say_values[j]

        # we return the value of the unique prediction which had the highest amount_of_say.
        return unique_predictions[amount_of_say_per_unique_prediction.index(max(amount_of_say_per_unique_prediction))]

    def predict(self, x_test):
        """
        :brief: This function calculates the predictions for each row of the test set.

        :param x_test: is the test set of which we will predict its values.
        :return: the predictions of our AdaBoost classifier.
        """

        return [self.classify(x) for x in x_test]
