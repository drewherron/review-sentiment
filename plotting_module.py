import matplotlib.pyplot as plt
import numpy as np


# Plot the model metrics
def plot_results(results, title='Model Performance'):
    keys_for_epochs = ['training_loss', 'validation_loss', 'training_accuracy', 'validation_accuracy']
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 20

    # Plot with epochs or with classes depending on model results.
    if any(key in results for key in keys_for_epochs):
        # Plot training loss (optional)
        if 'training_loss' in results:
            plt.plot(range(len(results['training_loss'])), results['training_loss'], label='Training Loss', color='orange', linestyle='dashed')

        # Plot testing loss (optional)
        if 'validation_loss' in results:
            plt.plot(range(len(results['validation_loss'])), results['validation_loss'], label='Testing Loss', color='green', linestyle='dashed')

        # Plot training accuracy
        if 'training_accuracy' in results:
            plt.plot(range(len(results['training_accuracy'])), results['training_accuracy'], label='Training Accuracy', color='red')

        # Plot validation accuracy
        if 'validation_accuracy' in results:
            plt.plot(range(len(results['validation_accuracy'])), results['validation_accuracy'], label='Testing Accuracy', color='blue')

        plt.xlabel('Epochs')

    else:
        # Plot testing precision
        if 'testing_precisions' in results:
            plt.plot(range(1, len(results['testing_precisions']) + 1), results['testing_precisions'], label='Precision',
                     color='orange', linestyle='dashed')

        # Plot testing recall
        if 'testing_recalls' in results:
            plt.plot(range(1, len(results['testing_recalls']) + 1), results['testing_recalls'], label='Recall',
                     color='green', linestyle='dashed')

        # Plot training accuracy
        if 'testing_f1_scores' in results:
            plt.plot(range(1, len(results['testing_f1_scores']) + 1), results['testing_f1_scores'], label='F1 Score',
                     color='red')

        plt.xlabel('Class')

    plt.ylabel('Metrics')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

dummy_data = {"training_loss": [1.610927939414978, 1.3955731391906738, 1.0948947668075562, 0.8704226016998291, 0.5764394402503967, 0.37585246562957764, 0.21852800250053406, 0.13136914372444153, 0.124045230448246, 0.10272974520921707], "training_accuracy": [0.23125000298023224, 0.41874998807907104, 0.5270833373069763, 0.65625, 0.8208333253860474, 0.9104166626930237, 0.949999988079071, 0.9833333492279053, 0.9666666388511658, 0.9833333492279053], "validation_loss": [1.507208228111267, 1.2635241746902466, 1.2019901084899902, 1.1257754564285278, 1.050043249130249, 0.8394574165344238, 0.7696824312210083, 0.5421780347824097, 0.564292073249817, 0.7187234163284302], "validation_accuracy": [0.3583333194255829, 0.44999998807907106, 0.4533333373069805, 0.46666666865348816, 0.5416666865348816, 0.56666666865348816, 0.6583333373069763, 0.69166667461395264, 0.7149999940395355, 0.79166667461395264], "testing_loss": 1.7187234163284302, "testing_accuracy": 0.49166667461395264, "confusion_matrix": [[11, 10, 1, 3, 0], [4, 14, 7, 3, 0], [0, 10, 9, 2, 0], [0, 0, 3, 13, 4], [0, 1, 2, 11, 12]]}
