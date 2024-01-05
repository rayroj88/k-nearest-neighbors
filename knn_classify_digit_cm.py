import numpy as np
from get_features_cm import get_features_cm

def knn_classify_digit_cm(digit, K, train_cmoments_db):
    ''' Classify a digit using the KNN classifier based on central moments. '''

    # Extract training data
    train_cmoments, train_labels, min_vals, max_vals = train_cmoments_db

    # Compute the central moments for the query digit using the get_features_cm function
    query_moments = get_features_cm(digit)

    # Calculate the Euclidean distances between the query digit and all training digits
    distances = np.sqrt(np.sum((train_cmoments - query_moments) ** 2, axis=1))

    # Find the indices of the K-nearest neighbors
    nearest_indices = np.argsort(distances)[:K]

    # Get the labels of the K-nearest neighbors
    nearest_labels = train_labels[nearest_indices].astype(int)

    # Predict the label by finding the majority class among the nearest neighbors
    prediction = np.argmax(np.bincount(nearest_labels))

    return prediction