
import matplotlib.pyplot as plt
import numpy as np


# Plot the model metrics
def plot_results(results, title='Model Performance'):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 20

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
    plt.ylabel('Metrics')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

dummy_data = {"training_loss": [1.6197402477264404, 1.557472825050354, 1.442044973373413, 1.431334137916565, 1.337188720703125], "training_accuracy": [0.1428571492433548, 0.3571428656578064, 0.5, 0.4285714328289032, 0.7142857313156128], "validation_loss": [1.6472587585449219, 1.7014179229736328, 1.7272690534591675, 1.7290664911270142, 1.7428969144821167], "validation_accuracy": [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], "final_loss": 1.7428969144821167, "final_accuracy": 0.1666666716337204, "confusion_matrix": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 2, 0, 0, 0], [1, 1, 0, 0, 0]]}
