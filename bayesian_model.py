import nltk
import json
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Naive Bayes classification using MultinomialNB or NLTK Naive Bayes
# classifier.
# -----------------------------------------------------------------------------

# Selection menu for choosing model to use and number and type of features
# (Saves space in main file).
def get_model_parameters(test_size=1.0):
    choice = input("Select model\n1. Multinomial Naive Bayes classifier\n"
                   "2. NLTK Naive Bayes classifier\n>> ")
    if choice in ['1', '2']:
        if choice == '1':
            model_name = 'MultinomialNB_model'
        else:
            model_name = 'NLTK_NB_model'
    else:
        print("Invalid choice")
        return 0, 0, 0, 0

    # Test size not equal to 1 indicates model is not loaded.
    if test_size != 1.0:
        # Choose number of and type of features.
        choice = input("Use all unique words as features?\n1. Yes\n2. No\n>> ")
        if choice in ['1', '2']:
            feat_all = True
            n_features = 1
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
    return model_name


class BayesSentiment:
    def __init__(self, model_name='MultinomialNB_model', n_features=1000,
                 feat_all=False, is_meaningful=True, batch_size=1000):
        self.model_name = model_name
        self.model = self.compile_model()
        self.n_features = n_features
        self.feat_all = feat_all
        self.is_meaningful = is_meaningful
        self.batch_size = batch_size
        self.features = []

    def process_data(self, x_train, x_test):
        # Holds words with little meaning.
        stop_words = set(nltk.corpus.stopwords.words('english'))
        reviews = x_train + x_test

        # Gather all split words from each review that are not stop words.
        for review in tqdm(reviews, "Processing Features"):
            # Check for and skip reviews with NaN values.
            if not pd.isna(review):
                tokens = nltk.word_tokenize(review)
                filtered_words = [word.lower() for word in tokens
                                  if word.isalpha() and word.lower()
                                  not in stop_words]
                for tagged_words in nltk.pos_tag(filtered_words):
                    yield tagged_words

    # Select feature words, all or most frequent, all types or
    # adjective/adverbs only.
    def choose_features(self, x_train, x_test):
        # Holds words with little meaning.
        # stop_words = set(nltk.corpus.stopwords.words('english'))
        # tagged_words = []
        # reviews = x_train + x_test

        # Gather all split words from each review that are not stop words.
        # for review in tqdm(reviews, "Processing Features"):
            # Check for and skip reviews with NaN values.
            # if not pd.isna(review):
            #     tokens = nltk.word_tokenize(review)
            #     filtered_words = [word.lower() for word in tokens
            #                       if word.isalpha() and word.lower()
            #                       not in stop_words]
            #     tagged_words.extend(nltk.pos_tag(filtered_words))
        tagged_words = self.process_data(x_train, x_test)
        # Identify parts of speech for each word that relate to adjectives (JJ,
        # JJR, JJS) or adverbs (RB) - Meaningful words.
        if self.is_meaningful:
            split_words = [word for word, pos in tagged_words
                           if pos.startswith('JJ') or pos.startswith('JJR')
                           or pos.startswith('JJS') or pos.startswith('RB')]
        else:
            split_words = [word for word, pos in tagged_words]

        # Count the frequency of each word in the new array.
        freq_dist = nltk.probability.FreqDist(split_words)

        # Return the chosen features.
        if not self.feat_all:
            self.features = [word for word, pos in freq_dist.most_common(int(self.n_features))]
        else:
            self.features = list(set(split_words))

        print(len(self.features))

    # Get dataset in the right format for Pandas.
    def encode_examples(self, reviews):
        # Batch the data for memory issues.
        n_batches = len(reviews) // self.batch_size + 1
        batches = [reviews[i * self.batch_size:(i + 1) * self.batch_size]
                   for i in range(n_batches)]
        processed_data = []

        # Follow progress for each batch.
        for batch in tqdm(batches, "Encoding Examples"):
            for review in batch:
                if not pd.isna(review):
                    processed_data.append(self.track_word_occurrence(review))

        # Return correct format for correct model.
        if self.model_name == 'MultinomialNB_model':
            return pd.DataFrame(processed_data)
        return processed_data

    # Yield the word frequency counts for all reviews.
    # def track_word_wrapper(self, reviews):
    #     for review in reviews:
    #         if not pd.isna(review):
    #             yield self.track_word_occurrence(review)

    # Count number of feature occurrences in a review.
    def track_word_occurrence(self, review):
        words = nltk.word_tokenize(review.lower())
        if self.model_name == 'MultinomialNB_model':
            word_occurs = {word: words.count(word) for word in self.features}
        else:
            word_occurs = {word: word in words for word in self.features}
        return word_occurs

    # Compile model either MultinomialNB or NLTK classifier model.
    def compile_model(self):
        # NLTK requires the training dataset and will be compiled later if
        # chosen.
        if self.model_name == 'MultinomialNB_model':
            return MultinomialNB()

    # Train a model with the training dataset.
    def fit(self, train_dataset, train_labels):
        if self.model_name == 'MultinomialNB_model':
            self.model.fit(train_dataset, train_labels)
        else:
            labeled_dataset = list(zip(train_dataset, train_labels))
            self.model = nltk.classify.NaiveBayesClassifier.train(labeled_dataset)

    # Test and predict with respect to the testing dataset. Model needs to be
    # an argument due to NLTK restrictions.
    def predict(self, model, test_dataset):
        if self.model_name == 'MultinomialNB_model':
            return model.predict(test_dataset)
        else:
            return model.classify_many(test_dataset)

    # Evaluate a model's performance.
    def evaluate(self, actuals, predictions):
        accuracy = metrics.accuracy_score(actuals, predictions)
        loss = metrics.mean_squared_error(actuals, predictions)
        return loss, accuracy

    # Save the model.
    def save_features(self, save_path):
        # Store model and features together
        model_data = {'classifier': self.model, 'features': self.features}
        joblib.dump(model_data, save_path)

    # Load an existing model.
    def load_features(self, load_path):
        model_data = joblib.load(load_path)
        self.model = model_data['classifier']
        self.features = model_data['features']

    # Plot confusion matrix.
    def confusion_matrix(self, predictions, actuals):
        cm = np.zeros((5, 5), dtype=int)
        for prediction, actual in zip(predictions, actuals):
            cm[int(actual - 1)][int(prediction - 1)] += 1
        return cm

    # Test an existing model on a test dataset.
    def test(self, x_test, y_test):
        # Format testing dataset with respect to feature categories.
        test_dataset = self.encode_examples(x_test)

        # Test and get results
        predictions = self.predict(self.model, test_dataset)
        loss, accuracy = self.evaluate(y_test, predictions)

        results = {
            'testing_loss': loss,
            'testing_accuracy': accuracy,
            'confusion_matrix': self.confusion_matrix(predictions, y_test)
        }

        return results

    # Format dataset, train model, then test.
    def train_and_test(self, x_train, x_test, y_train, y_test):
        # Process features
        self.choose_features(x_train, x_test)

        # Format training and testing dataset with respect to feature
        # categories.
        train_dataset = self.encode_examples(x_train)
        test_dataset = self.encode_examples(x_test)

        # Fit training dataset, test, and get results.
        self.fit(train_dataset, y_train)
        predictions = self.predict(self.model, test_dataset)
        loss, accuracy = self.evaluate(y_test, predictions)

        results = {
            'testing_loss': loss,
            'testing_accuracy': accuracy,
            'confusion_matrix': self.confusion_matrix(predictions, y_test)
        }

        return results
