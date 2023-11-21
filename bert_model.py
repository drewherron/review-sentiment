import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import os
# Filter out Tensorflow messages (set to 0 to see all)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class BertSentiment:

    def __init__(self, model_name='bert-base-uncased', num_labels=5, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.max_length = max_length

    # Get datset in the right format for Tensorflow
    def encode_examples(self, texts, labels):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []

        for text, label in zip(texts, labels):
            bert_input = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,     # Add [CLS] and [SEP]
                max_length=self.max_length,  # Use class attribute for max_length
                padding='max_length',        # Pad sentence to max length
                truncation=True,             # Truncate longer messages
                return_attention_mask=True,  # Return attention mask
            )

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append(label)

        # Create a TensorFlow dataset from the encoded data
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': input_ids_list,
                'token_type_ids': token_type_ids_list,
                'attention_mask': attention_mask_list
            },
            label_list
        ))

        return dataset


    # Compile the BERT model
    def compile_model(self, learning_rate=2e-5):

        # Using Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

        # Sparse Categorical Crossentropy
        # logits: not softmax/probability labels
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Use Sparse Categorical Accuracy as metric
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


    # Test the model
    # (but simpler, not to call from main)
    def evaluate(self, test_dataset, batch_size=8):

        # Batch the test dataset
        test_dataset = test_dataset.batch(batch_size)

        # Evaluate the model
        loss, accuracy = self.model.evaluate(test_dataset)

        return loss, accuracy


    # Save model to file
    def save_model(self, save_path):

        # Save the trained model
        self.model.save(save_path, save_format='tf')


    # Load model from file
    def load_model(self, model_path):

        # Load a trained model
        self.model = tf.keras.models.load_model(model_path)
        # to load a model:
        #classifier = BertSentiment()
        #classifier.load_model('path/to/saved/model')


    # Plot confusion matrix
    def confusion_matrix(self, predictions, actuals):

        cm = np.zeros((5, 5), dtype=int)
        for prediction, actual in zip(predictions, actuals):
            cm[int(actual)][int(prediction)] += 1

        return cm


    # Test the previously trained and loaded model
    def test(self, x_test, y_test, print_cm=False, batch_size=8):
        # Adjust the labels to be zero-indexed
        y_test = [label - 1 for label in y_test]

        # Encode the testing data
        test_dataset = self.encode_examples(x_test, y_test)
        test_dataset_batched = test_dataset.batch(batch_size)

        # Evaluate the model
        print("\nTesting model...")
        loss, accuracy = self.model.evaluate(test_dataset_batched)

        # Get predictions
        print("\nGetting predictions...")
        raw_predictions = self.model.predict(test_dataset_batched)
        predicted_labels = np.argmax(raw_predictions['logits'], axis=1)

        # Calculate fuzzy accuracy (within one)
        within_one = self.fuzzy_accuracy(predicted_labels, y_test)
        print(f"Adjusted Accuracy (Within One): {within_one}")

        # Results
        results = {
            'testing_loss': loss,
            'testing_accuracy': accuracy,
            'within_one': within_one
        }

        # Add confusion matrix
        if print_cm:
            print("\nGenerating confusion matrix...")
            cm = self.confusion_matrix(predicted_labels, y_test)
            results['confusion_matrix'] = cm.tolist()

        return results


    # Get accuracy within one star difference
    def fuzzy_accuracy(self, predictions, actuals):
        correct = 0
        for pred, actual in zip(predictions, actuals):
            # Prediction is within one star of actual rating
            if abs(pred - actual) <= 1:
                correct += 1
        return correct / len(predictions)

    # Train the model, then test
    def train_and_test(self, x_train, x_test, y_train, y_test, print_cm=False, epochs=4, batch_size=8):

        # Adjust the labels to be zero-indexed
        y_train = [label - 1 for label in y_train]
        y_test = [label - 1 for label in y_test]

        # Encode the training and testing data
        train_dataset = self.encode_examples(x_train, y_train)
        test_dataset = self.encode_examples(x_test, y_test)

        # Compile the model
        self.compile_model(learning_rate=2e-5)

        # Train the model
        print("\nTraining model...")
        history = self.model.fit(
            train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE),
            epochs=epochs,
            validation_data=test_dataset.batch(batch_size)
        )

        # Test the model
        print("\nTesting model...")
        loss, accuracy = self.model.evaluate(test_dataset.batch(batch_size))

        # Results for plotting
        results = {
            'training_loss': history.history['loss'],
            'training_accuracy': history.history['accuracy'],
            'validation_loss': history.history['val_loss'],
            'validation_accuracy': history.history['val_accuracy'],
            'testing_loss': loss,
            'testing_accuracy': accuracy
        }

        # Confusion matrix
        if print_cm:
            # Get predictions
            print("\nGetting predictions for confusion matrix...")
            test_dataset_batched = test_dataset.batch(batch_size)
            raw_predictions = self.model.predict(test_dataset_batched)
            predicted_labels = np.argmax(raw_predictions.logits, axis=1)

            # Create confusion matrix
            cm = self.confusion_matrix(predicted_labels, y_test)

            # Update results with confusion matrix data
            results['confusion_matrix'] = cm.tolist()

        return results
