import pandas as pd
import gzip
import json
import nltk

# Needed to download NLTK prior to running this test program, then I commented
# it out.
# nltk.download()

# For some reason, I put the data in a different folder. Used the Electronics
# reviews.
filename = './data/Electronics.json.gz'


# Read the data into a pandas data frame with parse and getDF (code from
# dataset source: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ ).
def parse(path):
    count = 0
    g = gzip.open(path, 'rb')
    for line in g:
        count += 1
        # Messy break as to not deal with all the data right now.
        if count > 1000:
            break
        yield json.loads(line)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# This function uses the NLTK library to split, find, and display the top ten
# "meaningful words" from the given dataset, which, in this case, are
# adjectives and adverbs.
def count_top_words(dataset, num_words=10):
    rows = dataset.shape[0]
    tagged_words = []
    # Holds insignificant words with little meaning.
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Gather all split words from each review that are not stop words.
    for i in range(rows):
        review = dataset.loc[i, 'reviewText']

        # Some reviews had NaN values, which disrupted the program, so this
        # checks for that.
        if not pd.isna(review):
            tokens = nltk.word_tokenize(dataset.loc[i, 'reviewText'])
            filtered_words = [word.lower() for word in tokens
                              if word.isalpha() and word.lower()
                              not in stop_words]
            tagged_words.extend(nltk.pos_tag(filtered_words))

    # Identify parts of speech for each word that relate to adjectives (JJ, JJR, JJS) or adverbs (RB).
    meaningful_words = [word for word, pos in tagged_words
                        if pos.startswith('JJ') or pos.startswith('JJR')
                        or pos.startswith('JJS') or pos.startswith('RB')]

    # Counts the frequency of each word in the new array.
    freq_dist = nltk.probability.FreqDist(meaningful_words)

    # Holds the top ten most frequent words in sorted order (most to least).
    top_words = freq_dist.most_common(num_words)

    return top_words


# Run test of NLTK
data = get_df(filename)
print(data.loc[:, 'reviewText'])
count_top_words(data)
