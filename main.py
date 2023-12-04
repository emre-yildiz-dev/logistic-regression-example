import numpy as np


from typing import Tuple

def generate_synthetic_data(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset for logistic regression.

    Parameters:
    num_samples (int): The number of samples to generate

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the features and labels of the dataset
    """
    # Randomly generating features
    feature1 = np.random.randn(num_samples) * 10  # Feature 1
    feature2 = np.random.randn(num_samples) * 10  # Feature 2

    # Generating labels based on a simple rule
    # For example, positive class if feature1 - feature2 > threshold, else negative class
    threshold = 5
    labels = np.where((feature1 - feature2) > threshold, 1, 0)

    # Combining the features into a single array
    features = np.column_stack((feature1, feature2))

    return features, labels
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid function.

    Parameters:
    z (np.ndarray): The input array or value

    Returns:
    np.ndarray: Sigmoid of z
    """
    return 1 / (1 + np.exp(-z))

def predict_logistic_regression(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Predicts the probability of the samples belonging to the positive class.

    Parameters:
    X (np.ndarray): The input features.
    weights (np.ndarray): The weights of the logistic regression model.

    Returns:
    np.ndarray: The predicted probabilities.
    """
    # Linear combination of weights and features
    z = np.dot(X, weights)
    # Applying the sigmoid function
    predictions = sigmoid(z)
    return predictions

def logistic_regression_cost(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the binary cross-entropy cost for logistic regression.

    Parameters:
    y_true (np.ndarray): The true labels.
    predictions (np.ndarray): The predicted probabilities.

    Returns:
    float: The calculated cost.
    """
    m = y_true.shape[0]  # Number of samples
    # Binary cross-entropy cost
    cost = -1/m * np.sum(y_true * np.log(predictions) + (1 - y_true) * np.log(1 - predictions))
    return cost

def gradient_descent(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    Performs gradient descent to update the weights of the logistic regression model.

    Parameters:
    X (np.ndarray): The input features.
    y (np.ndarray): The true labels.
    weights (np.ndarray): The current weights of the model.
    learning_rate (float): The learning rate for weight updates.

    Returns:
    np.ndarray: The updated weights.
    """
    m = X.shape[0]  # Number of samples
    predictions = predict_logistic_regression(X, weights)
    # Gradient of the cost function
    gradient = np.dot(X.T, (predictions - y)) / m
    # Update the weights
    weights -= learning_rate * gradient
    return weights

def train_logistic_regression(X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> Tuple[np.ndarray, list]:
    """
    Trains a logistic regression model using gradient descent.

    Parameters:
    X (np.ndarray): The input features.
    y (np.ndarray): The true labels.
    epochs (int): The number of iterations over the entire dataset.
    learning_rate (float): The learning rate for weight updates.

    Returns:
    Tuple[np.ndarray, list]: The trained weights and the list of costs for each epoch.
    """
    weights = np.zeros(X.shape[1])  # Initializing weights to zeros
    costs = []  # To store the cost of each epoch

    for epoch in range(epochs):
        # Making predictions
        predictions = predict_logistic_regression(X, weights)
        # Calculating cost
        cost = logistic_regression_cost(y, predictions)
        costs.append(cost)
        # Updating weights using gradient descent
        weights = gradient_descent(X, y, weights, learning_rate)

        # (Optional) Print cost every certain number of epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Cost: {cost}")

    return weights, costs


def manual_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Manually splits the dataset into training and testing sets.

    Parameters:
    X (np.ndarray): The input features.
    y (np.ndarray): The true labels.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The training and testing sets (features and labels).
    """
    # Determine the split index
    split_index = int(X.shape[0] * (1 - test_size))

    # Split the dataset
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

def manual_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the accuracy score.

    Parameters:
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted labels.

    Returns:
    float: The accuracy score.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def manual_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the confusion matrix.

    Parameters:
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted labels.

    Returns:
    np.ndarray: The confusion matrix.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Generating the synthetic dataset
    num_samples = 100  # Number of samples in the dataset
    features, labels = generate_synthetic_data(num_samples)

    # Example usage of gradient descent
    # Initialize weights
    weights = np.zeros(features.shape[1])

    # Perform one iteration of gradient descent
    learning_rate = 0.01
    weights = gradient_descent(features, labels, weights, learning_rate)
    print(weights)  # Displaying the updated weights for review
    # Training the model
    epochs = 100  # Number of epochs for training
    learning_rate = 0.01  # Learning rate
    trained_weights, training_costs = train_logistic_regression(features, labels, epochs, learning_rate)

    print(trained_weights)  # Displaying the trained weights

    # Splitting the dataset manually
    X_train, X_test, y_train, y_test = manual_train_test_split(features, labels, test_size=0.2)

    # Training the model on the training set
    trained_weights, _ = train_logistic_regression(X_train, y_train, epochs, learning_rate)

    # Making predictions on the test set
    test_predictions_prob = predict_logistic_regression(X_test, trained_weights)
    test_predictions = (test_predictions_prob > 0.5).astype(int)  # Converting probabilities to binary predictions

    # Evaluating the model manually
    accuracy = manual_accuracy_score(y_test, test_predictions)
    conf_matrix = manual_confusion_matrix(y_test, test_predictions)

    print(accuracy, conf_matrix)  # Displaying the accuracy and confusion matrix

