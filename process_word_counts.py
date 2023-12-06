import pandas as pd
import nltk
from tqdm import tqdm

# Download NLTK elements prior to running this test program.
# nltk.download()


# Choose dataset-wide features to be used for each review's frequency counts.
def choose_features(x_train, x_test, n_features, all_words=False,
                    is_meaningful=True):
    # Holds words with little meaning.
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tagged_words = []
    reviews = x_train + x_test

    # Gather all split words from each review that are not stop words.
    for review in reviews:
        # Check for and skip reviews with NaN values.
        if not pd.isna(review):
            tokens = nltk.word_tokenize(review)
            filtered_words = [word.lower() for word in tokens
                              if word.isalpha() and word.lower()
                              not in stop_words]
            tagged_words.extend(nltk.pos_tag(filtered_words))

    # Identify parts of speech for each word that relate to adjectives (JJ,
    # JJR, JJS) or adverbs (RB) - Meaningful words.
    if is_meaningful:
        split_words = [word for word, pos in tagged_words
                       if pos.startswith('JJ') or pos.startswith('JJR')
                       or pos.startswith('JJS') or pos.startswith('RB')]
    else:
        split_words = [word for word, pos in tagged_words]

    # Count the frequency of each word in the new array.
    freq_dist = nltk.probability.FreqDist(split_words)

    # Return the chosen features.
    if not all_words:
        features = [word for word, pos in freq_dist.most_common(int(n_features))]
    else:
        features = list(set(split_words))

    # print(features)
    return features


# Get dataset in the right format for Pandas.
def encode_examples(reviews, features, batch_size=1000):
    # Batch the data for memory issues.
    n_batches = len(reviews) // batch_size + 1
    batches = [reviews[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
    processed_data = []

    # Follow progress for each batch.
    for batch in tqdm(batches, "Processing Batches"):
        processed_data.extend(word_freq_wrapper(batch, features))

    dataset = pd.DataFrame(processed_data)
    return dataset


# Count number of feature occurrences in a review.
def count_word_freq(review, features):
    words = nltk.word_tokenize(review.lower())
    word_counts = {word: words.count(word) for word in features}
    return word_counts


# Yield the word frequency counts for all reviews.
def word_freq_wrapper(reviews, features):
    for review in reviews:
        if not pd.isna(review):
            yield count_word_freq(review, features)
