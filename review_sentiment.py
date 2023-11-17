import json
import random
import argparse
from sklearn.model_selection import train_test_split

#import nn_model
#import bayesian_model
#import bert_model
#import gpt_model
#import plot_module


# These are defaults, overridden by command line arguments
FILE_PATH = "./data/Appliances.json"
MAX_REVIEWS = 200000   # Max number of reviews to load from data
TEST_SIZE = 0.2        # Percentage of data to reserve for testing
SEED = None            # Seed the randomness of the initial data shuffles


# Allow command line arguments
def get_args():
    parser = argparse.ArgumentParser(
        prog="python3 review_sentiment.py",
        description="Classifying Amazon reviews.")
    parser.add_argument("-b", "--balanced",
                        action="store_true",
                        help="load balanced proportion of ratings")
    parser.add_argument("-f", "--file",
                        type=str,
                        default=FILE_PATH,
                        help="data file to be used as input")
    parser.add_argument("-n", "--num-reviews",
                        type=int,
                        default=MAX_REVIEWS,
                        help="max number of reviews to read from data")
    parser.add_argument("-s", "--seed",
                        type=int,
                        default=SEED,
                        help="seed for shuffling the data at loading")
    parser.add_argument("-t", "--test",
                        type=float,
                        default=TEST_SIZE,
                        metavar="TESTING_PERCENTAGE",
                        help="percentage of data reserved for testing")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="increase output verbosity.")
    return parser.parse_args()


# Load data
def load_data(filename, max_reviews, test_size, seed):

    reviews = []
    ratings = []

    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= max_reviews:
                break
            try:
                review = json.loads(line)
                reviews.append(review.get('reviewText', ''))
                ratings.append(review.get('overall', 0))
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {i+1}")

    # Split the data into training and testing sets
    train_in, test_in, train_tgt, test_tgt = train_test_split(reviews, ratings, test_size=test_size, random_state=seed, shuffle=True)

    return train_in, test_in, train_tgt, test_tgt


# Load the same number of reviews from each rating level
def load_balanced_data(filename, max_reviews, test_size, seed):

    reviews_per_rating = {1.0: [], 2.0: [], 3.0: [], 4.0: [], 5.0: []}
    all_reviews = []

    # Read in all reviews
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                review = json.loads(line)
                all_reviews.append((review.get('reviewText', ''), review.get('overall', 0)))
            except json.JSONDecodeError:
                print(f"Error decoding JSON")

    # Shuffle all reviews
    random.seed(seed)
    random.shuffle(all_reviews)

    # Split reviews into 5 rating categories
    max_reviews_per_rating = max_reviews // 5
    for review, rating in all_reviews:
        if rating in reviews_per_rating and len(reviews_per_rating[rating]) < max_reviews_per_rating:
            reviews_per_rating[rating].append((review, rating))

    # Find the minimum number of reviews in any rating category
    min_reviews = min(len(reviews) for reviews in reviews_per_rating.values())

    # Limit each category to the size of the smallest category
    for rating in reviews_per_rating:
        reviews_per_rating[rating] = reviews_per_rating[rating][:min_reviews]

    # Flatten the reviews and ratings
    reviews, ratings = zip(*[item for sublist in reviews_per_rating.values() for item in sublist])

    # Split the data into training and testing sets
    train_in, test_in, train_tgt, test_tgt = train_test_split(reviews, ratings, test_size=test_size, random_state=seed, shuffle=True)

    return train_in, test_in, train_tgt, test_tgt


def main():

    # Command-line arguments
    args = get_args()
    balanced_load = args.balanced
    max_reviews = args.num_reviews
    seed = args.seed
    test_size = args.test
    verbose = args.verbose
    file_path = args.file

    # Main menu
    choice = input("Select model:\n1. NN\n2. Bayesian\n3. BERT\n4. GPT\n5. Sanity check\nChoice: ")

    # Only load the data if the choice is valid (to save on time/computation)
    if choice in ['1', '2', '3', '4', '5']:
        if balanced_load:
            if verbose:
                print("Running load_balanced_data")
            train_in, test_in, train_tgt, test_tgt = load_balanced_data(file_path, max_reviews, test_size, seed)
        else:
            if verbose:
                print("Running load_data")
            train_in, test_in, train_tgt, test_tgt = load_data(file_path, max_reviews, test_size, seed)

    # Run the selected model
    # These could return dictionaries
    if choice == '1':
        results = nn_model.train_and_test(train_in, test_in, train_tgt, test_tgt)
    elif choice == '2':
        results = bayesian_model.train_and_test(train_in, test_in, train_tgt, test_tgt)
    elif choice == '3':
        results = bert_model.train_and_test(train_in, test_in, train_tgt, test_tgt)
    elif choice == '4':
        results = gpt_model.train_and_test(train_in, test_in, train_tgt, test_tgt)

    # Just for testing, remove before submission
    elif choice == '5':

        print("\nTesting\n-------\n")
        print(f"file_path:\t{file_path}\n")

        print(f"balanced_load:\t{balanced_load}")
        print(f"verbose:\t{verbose}")
        print(f"max_reviews:\t{max_reviews}")
        print(f"test_size:\t{test_size}\n")

        print(f"Number of training inputs: {len(train_in)}")
        print(f"Number of training targets: {len(train_tgt)}")
        print(f"Number of testing inputs: {len(test_in)}")
        print(f"Number of testing targets: {len(test_tgt)}\n")

        print(f"First training input: {train_in[0]}")
        print(f"First training target: {train_tgt[0]}")
        print(f"First testing input: {test_in[0]}")
        print(f"First testing target: {test_tgt[0]}")

    else:
        print("Invalid choice.")
        return

    # Plot results
    #plot_module.plot_results(results)

if __name__ == "__main__":
    main()
