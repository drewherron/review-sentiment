from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import process_word_counts

# Temporary notes: Using MultinomialNB unless I find something better. GaussianNB
#                  doesn't like the low frequencies, so counts are discrete rather
#                  than normalized distribution. Tensorflow is not right for simple
#                  Bayes. Possible NLTK Bayes classifier to use.

# Will transform into class to match other files.
# class BayesSentiment:
#     def __init__(self, model_name='Bayes_model', n_features=1000, feat_all=False, is_meaningful=True, batch_size=1000):
#         self.model = MultinomialNB()
#         self.model_name = model_name
#         self.n_features = n_features
#         self.feat_all = feat_all
#         self.is_meaningful = is_meaningful
#         self.batch_size = batch_size
#
#     def choose_features(self, x_train, x_test):
#         return process_word_counts.choose_features(x_train, x_test, self.n_features, self.feat_all, self.is_meaningful)
#
#     def preprocess_data(self, texts, features):
#         return process_word_counts.encode_examples(texts, features, self.batch_size)
#
#     def fit(self, train_dataset, train_labels):
#         self.model.fit(train_dataset, train_labels)
#
#     def predict(self, test_dataset):
#         return self.model.predict(test_dataset)
#
#     def evaluate(self, actuals, predictions):
#         accuracy = metrics.accuracy_score(actuals, predictions)
#         loss = metrics.log_loss(actuals, predictions)
#         return loss, accuracy
#
#     def save_model(self, save_path):
#         pass
#
#     def load_model(self, model_path):
#         pass
#
    # Plot confusion matrix
    # def confusion_matrix(self, predictions, actuals):
    #
    #     cm = np.zeros((5, 5), dtype=int)
    #     for prediction, actual in zip(predictions, actuals):
    #         cm[int(actual)][int(prediction)] += 1
    #     return cm
    #
    # def test(self, test_dataset, test_labels):
    #     pass


# Format dataset, train model, then test.
def train_and_test(x_train, x_test, y_train, y_test):
    all_words = True       # Whether to use all unique words as features
    is_meaningful = False  # Whether to use adjective/adverbs as features
    n_features = 10        # Number of features

    # Will move menu to review sentiment until Bayesian choice. Invalid choices
    # will just return and close program rather than how it is now.
    # Menu options for Bayes classification.
    choice = input("Use all unique words as features?\n1. Yes\n2. No\n>> ")
    if choice.upper() in ['1', '2']:
        if choice.upper() == '2':
            all_words = False
            try:
                sub_choice = int(input("Enter number of features (# of most "
                                       "common words - will max out at # of "
                                       "unique words):\n>> "))
                if sub_choice > 0:
                    n_features = sub_choice
                else:
                    print(f"Invalid choice. Default set to {n_features} "
                          f"features.")
            except ValueError as e:
                print(f"{e}. Default set to {n_features} features.")

        sub_choice = input("Select feature types:\n1. All Words\n"
                           "2. Adjectives/Adverbs Only\n>> ")
        if sub_choice in ['1', '2']:
            if sub_choice == '2':
                is_meaningful = True
        else:
            print(f"Invalid choice. Default set to Most Common Words.")
    else:
        print(f"Invalid choice. Default set to all unique words.")

    print("Processing data...")
    features = process_word_counts.choose_features(x_train, x_test, n_features,
                                                   all_words, is_meaningful)
    # Format training and testing dataset
    train_dataset = process_word_counts.encode_examples(x_train, features)
    test_dataset = process_word_counts.encode_examples(x_test, features)
    # print(train_dataset)
    # print(test_dataset)

    # Fit training dataset, test, and get results.
    model = MultinomialNB()
    model.fit(train_dataset, y_train)
    test_predictions = model.predict(test_dataset)
    accuracy = metrics.accuracy_score(y_test, test_predictions)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions, pos_label=1)

    # Still need to reformat file/ format return results/ fix bugs
    print(f"Accuracy: {accuracy}")
    # print(f"AUC: {0}".format(metrics.auc(fpr, tpr)))
    return 0
