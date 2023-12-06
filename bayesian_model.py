from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import process_word_counts


# -----------------------------------------------------------------------------
# Naive Bayes classification using MultinomialNB or NLTK Naive Bayes
# classifier.
# -----------------------------------------------------------------------------

# Selection menu for choosing model to use and number and type of features
# (Saves space in main file).
def get_model_parameters():
    choice = input("Select model\n1. MultinomialNB classifier\n2. NLTK classifier")
    if choice in ['1', '2']:
        if choice == '1':
            model_name = 'MultinomialNB_model'
        else:
            model_name = 'NLTK classifier'
    else:
        print("Invalid choice")
        return 0, 0, 0, 0

    # Choose number of and type of features.
    choice = input("Use all unique words as features?\n1. Yes\n2. No\n>> ")
    if choice in ['1', '2']:
        feat_all = True
        if choice == '2':
            feat_all = False
            try:
                sub_choice = int(input("Enter number of features (# of most "
                                       "common words - will max out at # of "
                                       "unique words):\n>> "))
                if sub_choice > 0:
                    n_features = sub_choice
                else:
                    print("Invalid choice.")
                    return 0, 0, 0, 0
            except ValueError as e:
                print(f"{e}")
                return 0, 0, 0, 0

        sub_choice = input("Select feature types:\n1. All Words\n"
                           "2. Adjectives/Adverbs Only\n>> ")
        if sub_choice in ['1', '2']:
            is_meaningful = False
            if sub_choice == '2':
                is_meaningful = True
        else:
            print("Invalid choice.")
            return 0, 0, 0, 0
    else:
        print("Invalid choice.")
        return 0, 0, 0, 0
    return model_name, feat_all, n_features, is_meaningful


class BayesSentiment:
    def __init__(self, model_name='MultinomialNB_model', n_features=1000,
                 feat_all=False, is_meaningful=True, batch_size=1000):
        self.model_name = model_name
        self.model = self.compile_model(model_name)
        self.n_features = n_features
        self.feat_all = feat_all
        self.is_meaningful = is_meaningful
        self.batch_size = batch_size

    # Select feature words, all or most frequent, all types or
    # adjective/adverbs only.
    def choose_features(self, x_train, x_test):
        return process_word_counts.choose_features(x_train, x_test,
                                                   self.n_features,
                                                   self.feat_all,
                                                   self.is_meaningful)

    # Encode data into readable format for model.
    def preprocess_data(self, texts, features):
        return process_word_counts.encode_examples(texts, features,
                                                   self.batch_size)

    # Compile model either MultinomialNB or NLTK classifier model.
    def compile_model(self, model_name):
        if model_name == 'MultinomialNB_model':
            return MultinomialNB()
        else:
            pass

    # Train a model with the training dataset.
    def fit(self, train_dataset, train_labels):
        self.model.fit(train_dataset, train_labels)

    # Test and predict with respect to the testing dataset.
    def predict(self, test_dataset):
        return self.model.predict(test_dataset)

    # Evaluate a model's performance.
    def evaluate(self, actuals, predictions):
        accuracy = metrics.accuracy_score(actuals, predictions)
        loss = metrics.mean_squared_error(actuals, predictions)
        return loss, accuracy

    # Save the model.
    def save_model(self, save_path):
        pass

    # Load an existing model.
    def load_model(self, model_path):
        pass

    # Plot confusion matrix.
    def confusion_matrix(self, predictions, actuals):
        cm = np.zeros((5, 5), dtype=int)
        for prediction, actual in zip(predictions, actuals):
            cm[int(actual)][int(prediction)] += 1
        return cm

    def test(self, test_dataset, test_labels):
        pass

    # Format dataset, train model, then test.
    def train_and_test(self, x_train, x_test, y_train, y_test, print_cm=False):
        # Process features
        features = self.choose_features(x_train, x_test)

        # Format training and testing dataset with respect to feature
        # categories.
        train_dataset = self.preprocess_data(x_train, features)
        test_dataset = self.preprocess_data(x_test, features)

        # Fit training dataset, test, and get results.
        self.fit(train_dataset, y_train)
        predictions = self.predict(test_dataset)
        loss, accuracy = self.evaluate(y_test, predictions)

        # Still need to reformat file/ format return results/ fix bugs
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        return 0
