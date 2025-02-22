'''
Start code for Project 1-Part 1 and optional 2. 
CSE583/EE552 PRML
LA: Mukhil Muruganantham, 2025
LA: Vedant Sawant, 2025
Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Md Sultan Mahmud
    PSU Email ID: mqm7099
    Description: 
        - ML.__init__: Initializes the Maximum Likelihood model and prepares a placeholder for storing the computed weights.
        - ML.fit: Fits the Maximum Likelihood model by computing the optimal weights using the closed-form solution for polynomial regression.
        - ML.predict: Predicts target values for input features based on the fitted weights and polynomial degree.
        - MAP.__init__: Initializes the Maximum A Posteriori model with the provided regularization parameters (alpha and beta) and prepares a placeholder for the weights.
        - MAP.fit: Fits the Maximum A Posteriori model by computing the weights using the MAP closed-form solution, including a regularization term to control overfitting.
        - MAP.predict: Predicts target values for input features using the fitted weights and polynomial degree.
}
'''


import math
import numpy as np

# TODO: write your MaximalLikelihood class 
class ML:
    def __init__(self):
        """
        Initializes the Maximum Likelihood model. Stores weights after fitting.
        """
        self.weights = None

    def fit(self, x, y, degree=0):
        """
        Fits the Maximum Likelihood model to the data.
        Args:
            x (np.array): The features of the data, shape (N,).
            y (np.array): The targets of the data, shape (N,).
            degree (int): The degree of the polynomial to fit.
        """
        # Construct the design matrix X based on the polynomial degree
        X = np.vander(x, N=degree + 1, increasing=True)
        
        # Compute the weights using the closed-form solution
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, x, degree=0):
        """
        Predicts the targets for the given input.
        Args:
            x (np.array): The features of the data, shape (N,).
            degree (int): The degree of the polynomial to use for prediction.
        Returns:
            np.array: The predicted target values.
        """
        # Construct the design matrix X based on the polynomial degree
        X = np.vander(x, N=degree + 1, increasing=True)
        
        # Compute the predictions
        return X @ self.weights


# TODO: write your MAP class
class MAP:
    def __init__(self, alpha=0.005, beta=11.1):
        """
        Initializes the Maximum A Posteriori model with regularization parameters.
        Args:
            alpha (float): The alpha parameter (regularization coefficient).
            beta (float): The beta parameter (noise precision).
        """
        self.alpha = alpha
        self.beta = beta
        self.weights = None

    def fit(self, x, y, degree=0):
        """
        Fits the Maximum A Posteriori model to the data.
        Args:
            x (np.array): The features of the data, shape (N,).
            y (np.array): The targets of the data, shape (N,).
            degree (int): The degree of the polynomial to fit.
        """
        # Construct the design matrix X based on the polynomial degree
        X = np.vander(x, N=degree + 1, increasing=True)
        
        # Compute the regularization term (αI)
        regularization_term = self.alpha * np.eye(X.shape[1])
        
        # Compute the weights using the MAP closed-form solution
        self.weights = np.linalg.inv(self.beta * (X.T @ X) + regularization_term) @ (self.beta * X.T @ y)

    def predict(self, x, degree=0):
        """
        Predicts the targets for the given input.
        Args:
            x (np.array): The features of the data, shape (N,).
            degree (int): The degree of the polynomial to use for prediction.
        Returns:
            np.array: The predicted target values.
        """
        # Construct the design matrix X based on the polynomial degree
        X = np.vander(x, N=degree + 1, increasing=True)
        
        # Compute the predictions as the product of X and weights
        return X @ self.weights
# End code for Project 1-Part 1 and optional 2.