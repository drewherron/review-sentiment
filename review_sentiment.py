import json
import argparse
import os
# Filter out Tensorflow messages (set to 0 to see all)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nn_model
import bayesian_model
import bert_model

import loading_module as ld
import plotting_module as pl


# These are defaults, overridden by command line arguments
IN_FILE_PATH = "./data/Appliances.json"
IN_FILE_PATH_TEST = None # Dataset to test (but not train) on
OUT_MODEL_PATH = None    # File path to save model
IN_MODEL_PATH = None     # File path to load model
JSON_RESULTS_PATH = None # File path to save JSON results
MAX_REVIEWS = 200000     # Max number of reviews to load from data
TEST_SIZE = 0.2          # Percentage of data to reserve for testing
EPOCHS = 4               # Number of epochs to run in training
SEED = None              # Seed the randomness of the initial data shuffles


# Allow command line arguments
def get_args():
    parser = argparse.ArgumentParser(
        prog="python3 review_sentiment.py",
        description="Classifying Amazon reviews.")
    parser.add_argument("-b", "--balanced",
                        action="store_true",
                        help="load balanced proportion of ratings")
    parser.add_argument("-c", "--confusion-matrix",
                        action="store_true",
                        help="print a confusion matrix after training/testing")
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=EPOCHS,
                        help="number of epochs to run in training")
    parser.add_argument("-d", "--data-file",
                        type=str,
                        default=IN_FILE_PATH,
                        help="data file to be used as input")
    parser.add_argument("-i", "--in-model",
                        type=str,
                        default=IN_MODEL_PATH,
                        help="file path to load model")
    parser.add_argument("-n", "--num-reviews",
                        type=int,
                        default=MAX_REVIEWS,
                        help="max number of reviews to read from data")
    parser.add_argument("-o", "--out-model",
                        type=str,
                        default=OUT_MODEL_PATH,
                        help="file path for saved model")
    parser.add_argument("-p", "--plot",
                        action="store_true",
                        help="plot results after testing")
    parser.add_argument("-r", "--results",
                        nargs='?',
                        const='training_results.json',
                        default=JSON_RESULTS_PATH,
                        help="JSON file for saving/loading results")
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


def main():

    # Command-line arguments
    args = get_args()
    balanced_load = args.balanced
    print_cm = args.confusion_matrix
    json_results_file = args.results
    max_reviews = args.num_reviews
    seed = args.seed
    test_size = args.test
    epochs = args.epochs
    verbose = args.verbose
    plot = args.plot
    data_file_path = args.data_file
    in_model_path = args.in_model
    out_model_path = args.out_model

    results = {}

    # Main menu
    choice = input("Select model:\n1. Neural network\n2. Bayesian model\n3. BERT\n4. Plot results\n>> ")

    # Load data
    # Only if the choice is valid (to save on time/computation)
    if choice in ['1', '2', '3', '4']:
        print("\nLoading data...")
        if balanced_load:
            if verbose:
                print("using load_balanced_data\n")
            x_train, x_test, y_train, y_test = ld.load_balanced_data(data_file_path, max_reviews, test_size, seed)
        else:
            if verbose:
                print("using load_data\n")
            x_train, x_test, y_train, y_test = ld.load_data(data_file_path, max_reviews, test_size, seed)

    if verbose:
        print(f"data_file_path:\t\t\t{data_file_path}")
        print(f"in_model_path:\t\t\t{in_model_path}")
        print(f"out_model_path:\t\t\t{out_model_path}\n")
        print(f"verbose:\t\t\t{verbose}")
        print(f"plot:\t\t\t\t{plot}")
        print(f"confusion_matrix:\t\t{print_cm}")
        print(f"json_results_file:\t\t{json_results_file}")
        print(f"balanced_load:\t\t\t{balanced_load}\n")
        print(f"max_reviews:\t\t\t{max_reviews}")
        print(f"test_size:\t\t\t{test_size}")
        print(f"epochs:\t\t\t\t{epochs}")
        print(f"seed:\t\t\t\t{seed}\n")

        print(f"Number of training inputs:\t{len(x_train)}")
        print(f"Number of training targets:\t{len(y_train)}")
        print(f"Number of testing inputs:\t{len(x_test)}")
        print(f"Number of testing targets:\t{len(y_test)}\n")

        #print(f"First training input:\t{x_train[0]}")
        #print(f"First training target:\t{y_train[0]}")
        #print(f"First testing input:\t{x_test[0]}")
        #print(f"First testing target:\t{y_test[0]}")

    # Run the selected model
    # MLP
    if choice == '1':
        model = nn_model.NNSentiment(learning_rate=0.0001, momentum=0.9, num_freq=200)
        final_loss, final_acc, confusion_matrix, results = model.train_and_test( x_train, y_train, x_test, y_test, 3)
        print("Final Accuracy: ", final_acc)
        print("Final Loss: ", final_loss)
        print("Confusion Matrix", confusion_matrix)
        print("Full results: ", results)

    # Bayesian
    elif choice == '2':

        if 0 < test_size < 1.0:

            if out_model_path is None:
                print("\nWARNING: No file path provided - model will not be saved.")

            # Choose model and feature parameters.
            model_name, n_features, feat_type = bayesian_model.get_model_parameters()

            if n_features != 0:
                # Instantiate Naive Bayes model.
                classifier = bayesian_model.BayesSentiment(model_name, n_features, feat_type)

                # Train, test, and get results.
                results = classifier.train_and_test(x_train, x_test, y_train, y_test)

            else:
                return

            # Save features used
            if out_model_path is not None:
                try:
                    classifier.save_features(out_model_path)
                except Exception as e:
                    print(f"Error: {e}")

            if verbose:
                print("Results:")
                print(results)

        elif test_size == 1.0:
            # Instantiate Naive Bayes model.
            classifier = bayesian_model.BayesSentiment()

            # Load features
            try:
                classifier.load_features(in_model_path)
                print("\nModel loaded and compiled successfully.\n")

                # Test the model
                results = classifier.test(x_test, y_test)

                if verbose:
                    print("Results:")
                    print(results)

            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice.")
            return

    # BERT
    elif choice == '3':

        # Instantiate BERT model
        classifier = bert_model.BertSentiment(num_labels=5, max_length=512, learning_rate=2e-5, batch_size=8)

        # Train BERT
        if 0 < test_size < 1.0:

            if out_model_path is None:
                print("\nWARNING: No file path provided - model will not be saved.")

            # Train and test the model
            results = classifier.train_and_test(x_train, x_test, y_train, y_test, print_cm, epochs)

            # Save trained model
            if out_model_path is not None:
                try:
                    classifier.save_model(out_model_path)
                except Exception as e:
                    print(f"Error: {e}")

            if verbose:
                print("Results:")
                print(results)

        # Test BERT
        elif test_size == 1.0:

            # No point in plotting a test
            plot = False

            # Load model
            try:
                classifier.load_model(in_model_path)
                print("\nModel loaded successfully.\n")

                # Not sure why it needed this
                classifier.compile_model()
                print("Model compiled successfully.")

                # Test the model
                results = classifier.test(x_test, y_test)

                if verbose:
                    print("Results:")
                    print(results)

            except Exception as e:
                print(f"Error: {e}")

        else:
            print("Invalid choice.")
            return

    # Plot results
    elif choice == '4':
        plot = True
        print_cm = True
        if json_results_file is None:
            json_results_file = input("Enter filename of saved results (or leave blank to test plot with dummy data):\n>> ").strip()

        if json_results_file:
            try:
                with open(json_results_file, 'r') as file:
                    results = json.load(file)
            except FileNotFoundError:
                print(f"File not found: {json_results_file}")
                return
            except json.JSONDecodeError:
                print(f"Error reading file: {json_results_file}. Be sure it's a valid JSON file.")
                return
        else:
            results = pl.dummy_data

    else:
        print("Invalid choice.")
        return

    # Plot results
    if plot:
        pl.plot_results(results)

    # Print results
    # TODO this needs to be changed if your models return a list of values
    # I'm using this to store a single value for the final result
    if 'testing_loss' in results:
        print(f"\nFinal testing loss:\t{results['testing_loss']}")
    if 'testing_accuracy' in results:
        print(f"Final testing accuracy:\t{results['testing_accuracy']}")
    if 'within_one' in results:
        print(f"Accuracy within 1 star:\t{results['within_one']}")
    if 'overall_precision' in results:
        print(f"Overall precision:\t{results['overall_precision']}")
    if 'overall_recall' in results:
        print(f"Overall recall:\t{results['overall_recall']}")
    if 'overall_f1_score' in results:
        print(f"Overall f1_score:\t{results['overall_f1_score']}")

    # Print confusion matrix
    if print_cm:
        if 'confusion_matrix' in results:
            print("\nConfusion matrix:")
            for row in results['confusion_matrix']:
                print(row)
    print()

    # Save results to JSON file
    if json_results_file is not None and choice != '4':
        with open(json_results_file, 'w') as file:
            json.dump(results, file)

        print(f"Results saved to {json_results_file}.\n")


if __name__ == "__main__":
    main()
