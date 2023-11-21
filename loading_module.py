import json
import random
from sklearn.model_selection import train_test_split


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
    x_train, x_test, y_train, y_test = train_test_split(reviews, ratings, test_size=test_size, random_state=seed, shuffle=True)

    return x_train, x_test, y_train, y_test


# Load the same number of reviews from each rating
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
    x_train, x_test, y_train, y_test = train_test_split(reviews, ratings, test_size=test_size, random_state=seed, shuffle=True)

    return x_train, x_test, y_train, y_test


# Use separate datasets for training and testing
def split_balanced_data(train_filename, test_filename, max_reviews, test_size, seed):
    def read_and_balance_data(filename, max_reviews_per_rating):
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
        for review, rating in all_reviews:
            if rating in reviews_per_rating and len(reviews_per_rating[rating]) < max_reviews_per_rating:
                reviews_per_rating[rating].append((review, rating))

        # Find the minimum number of reviews in any rating category
        min_reviews = min(len(reviews) for reviews in reviews_per_rating.values())

        # Limit each category to the size of the smallest category
        for rating in reviews_per_rating:
            reviews_per_rating[rating] = reviews_per_rating[rating][:min_reviews]

        # Flatten the reviews and ratings
        return zip(*[item for sublist in reviews_per_rating.values() for item in sublist])

    # Process training and testing data separately
    max_reviews_per_rating = max_reviews // 5
    x_train, y_train = read_and_balance_data(train_filename, max_reviews_per_rating)
    x_test, y_test = read_and_balance_data(test_filename, max_reviews_per_rating)

    return x_train, x_test, y_train, y_test
