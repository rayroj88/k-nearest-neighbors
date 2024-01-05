import os
import sys

# Get the absolute path of the script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

import numpy as np
import time
from sklearn.metrics import confusion_matrix
from get_features_cm import get_features_cm
from knn_classify_digit_cm import knn_classify_digit_cm

def main():
    data_file_path = os.path.join(current_directory, 'data', 'mnist_data.csv')
    output_dir = os.path.join(current_directory, 'output')

    ################# Load the data #################
    # Load the data. Use ',' as the delimiter
    data = np.loadtxt(data_file_path, delimiter=',')
    labels = data[:, 0]
    data = data[:, 1:]
    # Reshape the data to be a list of 28x28 2D images
    data = data.reshape(data.shape[0], 28, 28)

    # Split the data into training and test sets
    train_data = data[:5000]
    train_labels = labels[:5000]

    test_data = data[5000:6000]
    test_labels = labels[5000:6000]

    ########## Compute the central moments of training data ##########
    train_cmoments = np.zeros((train_data.shape[0], 8))
    for i in range(train_data.shape[0]):
        train_cmoments[i] = get_features_cm(train_data[i])

    # Scale all features of the database to the range [0, 1]
    min_vals = np.min(train_cmoments, axis=0)
    max_vals = np.max(train_cmoments, axis=0)
    train_cmoments = (train_cmoments - min_vals) / (max_vals - min_vals)

    train_cmoments_db = (train_cmoments, train_labels, min_vals, max_vals)
    

    ################# Classify test data #################
    # Predict the labels of all test digits

    # Time the prediction process
    start_time = time.time()

    predictions = np.zeros(len(test_labels))
    K = 11
    for i in range(len(test_labels)):
        # Predict the digit using the KNN classifier
        digit = test_data[i,:,:]
        predictions[i] = knn_classify_digit_cm(digit, K, train_cmoments_db)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('\nTime elapsed for prediction: {} seconds'.format(elapsed_time))

    # Calculate the accuracy
    correct = np.sum(predictions == test_labels)
    total = len(test_labels)
    accuracy = correct / total

    print('\nCorrectly predicted {} out of {} digits'.format(correct, total))
    print('Accuracy: {}'.format(accuracy))

    # Print confusion matrix
    print('\nConfusion matrix:')
    print(confusion_matrix(test_labels, predictions))

if __name__ == '__main__':
    main()