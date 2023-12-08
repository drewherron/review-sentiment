import nltk
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
# Model save path must have 'MNB_model' or 'NLTKnb_model' in the filename in
# order to load respective model.
# ------------------------------------------------------------------------------

# Selection menu for choosing model to use and number and type of features
# (Saves space in main file).
def get_model_parameters():
    choice = input("Select model\n1. Multinomial Naive Bayes classifier\n"
                   "2. NLTK Naive Bayes classifier\n>> ")
    if choice in ['1', '2']:
        if choice == '1':
            model_name = 'MNB_model'
        else:
            model_name = 'NLTKnb_model'
    else:
        print("Invalid choice")
        return 0, 0, 0

    # Choose number of and type of features.
    try:
        sub_choice = int(input("Enter number of features <= 1000 (# of most "
                               "common words):\n>> "))
        if 0 < sub_choice <= 1000:
            n_features = sub_choice
        else:
            print("Invalid choice.")
            return 0, 0, 0
    except ValueError as e:
        print(f"{e}")
        return 0, 0, 0

    sub_choice = input("Select feature types:\n1. Adjectives/Adverbs Only\n"
                       "2. Nouns/Verbs/Interjections Only\n"
                       "3. All of the above\n>> ")
    if sub_choice in ['1', '2', '3']:
        feat_type = sub_choice
    else:
        print("Invalid choice.")
        return 0, 0, 0

    return model_name, n_features, feat_type


class BayesSentiment:
    def __init__(self, model_name='MNB_model', n_features=1000,
                 feat_type='1', batch_size=1000):
        self.model_name = model_name
        self.model = self.compile_model()
        self.n_features = n_features
        self.feat_type = feat_type
        self.batch_size = batch_size
        self.features = []

    # Process words from training and testing dataset that are not stopwords or
    # stem words.
    def process_data(self, x_train, x_test):
        # Holds words with little meaning.
        stop_words = set(nltk.corpus.stopwords.words('english'))
        # reviews = x_train + x_test
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # Gather all split words from each review that are not stop words.
        for review in tqdm(x_train + x_test, "Processing Features"):
            # Check for and skip reviews with NaN values.
            if not pd.isna(review):
                tokens = nltk.word_tokenize(review)
                filtered_words = [
                    lemmatizer.lemmatize(word.lower()) for word in tokens
                    if word.isalpha() and word.lower() not in stop_words
                ]
                tagged_words = nltk.pos_tag(filtered_words)
                for word, pos in tagged_words:
                    yield word, pos

    # Return words that match the given part of speech tags.
    def get_word_pos(self, words, pos_tags):
        return [word for word, pos in words if pos in pos_tags]

    # Select feature words, all or most frequent, all types or
    # adjective/adverbs only.
    def choose_features(self, x_train, x_test):
        # Process non-stop and stemmed words.
        tagged_words = self.process_data(x_train, x_test)

        # Only count adjectives and adverbs.
        if self.feat_type == '1':
            pos_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']

        # Only count nouns, verbs, and interjections.
        elif self.feat_type == '2':
            pos_tags = [
                'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD',
                'VBG', 'VBN', 'VBP', 'VBZ', 'UH'
            ]

        # Count all of the above parts of speech.
        else:
            pos_tags = [
                'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN', 'NNS', 'NNP',
                'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'UH'
            ]

        split_words = self.get_word_pos(tagged_words, pos_tags)

        # Count the frequency of each word in the new array.
        freq_dist = nltk.probability.FreqDist(split_words)

        # Assign the most common words used as the chosen features.
        self.features = [word for word, pos in freq_dist.most_common(self.n_features)]

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
        if self.model_name == 'MNB_model':
            return pd.DataFrame(processed_data)
        return processed_data

    # Count number of feature occurrences in a review.
    def track_word_occurrence(self, review):
        words = nltk.word_tokenize(review.lower())
        if self.model_name == 'MNB_model':
            word_occurs = {word: words.count(word) for word in self.features}
        else:
            word_occurs = {word: word in words for word in self.features}
        return word_occurs

    # Compile model either MultinomialNB or NLTK classifier model.
    def compile_model(self):
        # NLTK requires the training dataset and will be compiled in fit
        # method, if chosen.
        if self.model_name == 'MNB_model':
            return MultinomialNB()

    # Train a model with the training dataset.
    def fit(self, train_dataset, train_labels):
        if self.model_name == 'MNB_model':
            self.model.fit(train_dataset, train_labels)
        else:
            labeled_dataset = list(zip(train_dataset, train_labels))
            self.model = nltk.classify.NaiveBayesClassifier.train(labeled_dataset)

    # Test and predict with respect to the testing dataset. Model needs to be
    # an argument due to NLTK restrictions.
    def predict(self, model, test_dataset):
        if self.model_name == 'MNB_model':
            return model.predict(test_dataset)
        else:
            return model.classify_many(test_dataset)

    # Evaluate a model's performance.
    def evaluate(self, actuals, predictions):
        accuracy = metrics.accuracy_score(actuals, predictions)
        precision = metrics.precision_score(actuals, predictions,
                                            average='weighted', zero_division=0)
        recall = metrics.recall_score(actuals, predictions,
                                      average='weighted', zero_division=0)
        f1_score = metrics.f1_score(actuals, predictions,
                                    average='weighted', zero_division=0)

        precisions = metrics.precision_score(actuals, predictions,
                                             average=None, zero_division=0)
        recalls = metrics.recall_score(actuals, predictions,
                                       average=None, zero_division=0)
        f1_scores = metrics.f1_score(actuals, predictions,
                                     average=None, zero_division=0)

        return (accuracy, precisions, recalls, f1_scores,
                precision, recall, f1_score)

    # Save the model.
    def save_features(self, save_path):
        # Store model and features together
        model_data = {'classifier': self.model, 'features': self.features}
        joblib.dump(model_data, save_path)

    # Load an existing model.
    def load_features(self, load_path):
        if 'MNB_model' in load_path:
            self.model_name = 'MNB_model'
        elif 'NLTKnb_model' in load_path:
            self.model_name = 'NLTKnb_model'
        else:
            raise Exception("Unrecognized model name of file path.")

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
        accuracy, precisions, recalls, f1_scores, precision, recall, f1_score = self.evaluate(y_test, predictions)

        results = {
            'testing_accuracy': accuracy,
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1_score': f1_score,
            'testing_precisions': precisions.tolist(),
            'testing_recalls': recalls.tolist(),
            'testing_f1_scores': f1_scores.tolist(),
            'confusion_matrix': self.confusion_matrix(predictions, y_test).tolist()
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
        accuracy, precisions, recalls, f1_scores, precision, recall, f1_score = self.evaluate(y_test, predictions)

        results = {
            'testing_accuracy': accuracy,
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1_score': f1_score,
            'testing_precisions': precisions.tolist(),
            'testing_recalls': recalls.tolist(),
            'testing_f1_scores': f1_scores.tolist(),
            'confusion_matrix': self.confusion_matrix(predictions, y_test).tolist()
        }

        return results
