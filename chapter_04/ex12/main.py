from zlib import crc32

import numpy as np
from sklearn.datasets import load_iris

"""
Exercise 12. Implement batch gradient descent with early stopping for softmax regression without using 
`Scikit-Learn`, only `NumPy`. Use it on a classification task such as the iris dataset.
"""

def main():
    # raw data set
    iris = load_iris(as_frame=True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris["target"].values
    # all possible target classes
    unique_classes = iris.target_names
    # supposed to print ['setosa' 'versicolor' 'virginica']
    print("There is a total of {} unique target classes for this dataset: {}".format(len(unique_classes), unique_classes))
    
    X_with_bias = add_bias_term(X)
    
    X_train, y_train, X_test, y_test = split_data_with_id_hash(X_with_bias, y, 0.2, random_state = 42)
    
    print("Train size:", X_train.shape, y_train.shape)
    print("Test size:", X_test.shape, y_test.shape)
    
    theta_optimal = batch_gradient_descent_for_softmax_with_early_stopping(X_train, len(unique_classes), y_train)
    
    test_logits = compute_logits(X_test, theta_optimal) # Compute raw scores for the test set
    test_probabilities = softmax_probabilities(test_logits) # Transform raw scores to probabilities
    test_predictions = np.argmax(test_probabilities, axis=1) # Predict class labels based on highest probability
    
    accuracy = np.mean(test_predictions == y_test)
    print(f"Model accuracy on test set is: {accuracy:.2f}")
  
def add_bias_term(X):  
    """
    Generally, a linear model makes predictions by computing a weighted sum of the input features, plus a
    constant called the bias term.
    
    Adds "1" as bias term feature to each element of the trainig set
    `numpy.c_` is used to perform feature augmentation
    `numpy.ones` creates ndarrays filled with 1
    
    Args:
      X - training set
    """
    return np.c_[np.ones(len(X)), X]

def split_data_with_id_hash(data, targets, test_ratio, random_state=None):
    """
    Splits the data into training, testing steps using hash-based IDs.
    
    Args:
      data: Input dataset (NumPy array or Pandas DataFrame)
      targets: Target values corresponding to the input data 
      test_ratio: Proportion of the dataset to be used as the test set
      randoom_state: Optional random seed for reproducibility
    """
    if random_state is not None:
      np.random.seed(random_state)
    else:
      np.random.seed(42)
      
    # generate synthetic IDs is not already present
    # numpy.arange creates an array of ints between 0 and n
    ids = np.arange(len(data))
    
    # hash-based splitting logic
    hashed_ids = np.array([crc32(np.int64(id_)) for id_ in ids])
    hashed_ids = hashed_ids % 2**32 # Ensure range for consistency
    
    test_threshold = int(test_ratio *2**32)
    
    in_test_set = hashed_ids < test_threshold
    in_training_set = ~in_test_set
    
    # split data and targets accordingly
    X_train, y_train = data[in_training_set], targets[in_training_set]
    X_test, y_test = data[in_test_set], targets[in_test_set]

    return X_train, y_train, X_test, y_test

def compute_logits(X, theta):
    """
    Compute logits for the input data. This function multiplies each row in X
    by the corresponding weights in theta (model's parameters). This produces raw scores, an
    intermediate step, before converting them into probabilities with softmax. 
    In softmax regression, the logits are computed directly as a linear combination of features and 
    weights. There is no additional activation function (like sigmoid) applied to logits, because 
    softmax itself handles the transformation to probabilities.
    
    Args:
      X: NumPy array of shape (n_samples, n_features), input data with bias term
      theta: NumPy array of shape (n_features, n_classes), model weights
      
    Returns:
      NumPy array of shape (n_samples, n_classes), the logits
    """
    return X @ theta

def one_hot_encoded(y, num_classes):
  """
  This function will try to mimic what sklearn.preprocessing.OnehotEncoder does -
  transform text targets, where for each string there will be a unique ordinal integer.
  
  Args:
    y: NumPy array of shape(n_samples,), target values
    num_classes: Integer, total number of classes
  """
  return np.eye(num_classes)[y]
  
def softmax_probabilities(logits):
    """
    Computes the softmax for each sample in the input matrix. First, it applies the exponential
    function to the logits. In this way, large scores become very large, and small scores
    become small but positive. To avoid numbers getting too big, it subtracts the maximum score
    in each row from all logits in that row. Finally, it normalizes the exponentials by dividing
    them by their sum for each row. This ensures that all probabilities for a single row add up
    to 1, e.g.: 0.6, 0.3, 0.1
    
    Args:
      logits: NumPy array of shape (n_samples, n_classes), the raw scores
      
    Returns:
      NumPy array of the same shape, containing probabilities 
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # Avoid numerical instability when computing large exponentials
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def batch_gradient_descent_for_softmax_with_early_stopping(X, num_classes, y, eta=0.1, n_epochs=1000, early_stop_tolerance=1e-5, patience=5):
    """
    Performs batch gradient descent to optimize theta for softmax regression.
    
    Args:
      X: NumPy array of shape (n_samples, n_features), input features (with bias term)
      y:  NumPy array of shape (n_samples,), target labels
      num_classes: Integer, total number of classes
      eta: Float, learning rate
      n_epochs: Integer, number of epochs
      early_stop_tolerance: Float, minimum relative loss improvement to continue training.
      patience: Integer, number of epochs to wait for improvement before stopping.
      
    Returns:
      theta: NumPy array of shape (n_features, n_classes), optimized weights
    """
    def cross_entropy_loss(y_true, y_pred):
        """
        cross-entropy loss rewards the model for being confident and correct while penalizing it for being wrong.
        cross-entropy measures how confident your predictions for the corrrect class
        
        np.log takes a tiny valie 1e-15 to avoid taking the log of 0, which is undefined
        """
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
    
    m, n = X.shape
    theta = np.random.randn(n, num_classes) # Randomly initialize weights based on the existing number of classes
    
    one_hot_labels = one_hot_encoded(y, num_classes)
    
    loss_history = [] # Keep track of loss values over epochs
    no_improvements_epocs = 0
    
    for epoch in range(n_epochs):
      logits = compute_logits(X, theta) # calculate the raw scores for each class
      probabilities = softmax_probabilities(logits) # converts scores into probabilities
      errors = probabilities - one_hot_labels # subtracts true labels from the probabilities. This shows how far off the predictions from correct answers.
      gradients = (X.T @ errors) / m # calculates how much each weight needs to change to reduce the error
      theta -= eta * gradients
      
      # calculate and record loss
      loss = cross_entropy_loss(one_hot_labels, probabilities)
      loss_history.append(loss)
      
      #Early stopping: monitor relative loss improvement
      if len(loss_history) > 1:
        relative_improvement = loss_history[-2] - loss
        if relative_improvement < early_stop_tolerance:
          no_improvements_epocs += 1
        else:
          no_improvements_epocs = 0
          
        if no_improvements_epocs >= patience:
          print(f"Early stopping at epoch {epoch}. No significant improvement for {patience} epochs.")
          break
          
      # Print loss every 100 epochs for monitoring
      if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    theta # return the trained weights

if __name__ == "__main__":
    main()
