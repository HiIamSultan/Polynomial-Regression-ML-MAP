'''
Start code for Project 1-Part 1: Linear Regression. 
CSE583/EE552 PRML
LA: Mukhil Muruganantham, 2025
LA: Vedant Sawant, 2025

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Md Sultan Mahmud
    PSU Email ID: mqm7099
    Description: 
        - generateNoisyData: Generates noisy sample points by adding Gaussian noise 
          to a sine function and saves the data in a .npz file.
        - plot_with_shadded_bar: Plots ground truth curves with shaded error bars, noisy data points, 
          and predictions from ML or MAP models. Saves the plots to the results directory, with 
          the model name, polynomial degree, and N (number of points) in the title and filename.
        - calculate_rmse: Computes the Root Mean Squared Error (RMSE) to evaluate the performance of 
          ML and MAP models by comparing predictions with true values.
        - linear_regression: Fits ML and MAP models to the data for polynomial degrees 3, 6, and 9. 
          Predicts values, calculates RMSE for each model, and visualizes the results with plots.
        - main: Takes N as a command-line argument, generates noisy data, visualizes the noisy data, 
          and runs the regression experiment for the given value of N.
}
'''


import math
import os

import matplotlib.pyplot as plt
import numpy as np
import argparse

# TODO: import your models
from models import ML, MAP

def generateNoisyData(num_points=50):
    """
    Generates noisy sample points and saves the data. The function will save the data as a npz file.
    Args:
        num_points: number of sample points to generate.
    """
    x = np.linspace(1, 4*math.pi, num_points)
    y = np.sin(x*0.5)

    # Define the noise model
    nmu = 0
    sigma = 0.3
    noise = nmu + sigma * np.random.randn(num_points)
    t = y + noise

    # Save the data
    np.savez('data.npz', x=x, y=y, t=t, sigma=sigma)

# Feel free to change aspects of this function to suit your needs.
# Such as the title, labels, colors, etc.
def plot_with_shadded_bar(x=None, y=None, sigma=None, predictions=None, model_name="", degree=None, N=None):
    """
    Plots the GT data for visualization and predictions.
    Args:
        x: x values
        y: y values
        sigma: standard deviation
        predictions: predicted values to be plotted.
        model_name: name of the model being plotted (e.g., ML or MAP).
        degree: polynomial degree used for fitting.
        N: number of data points used.
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    # Load data if not provided
    if x is None or y is None or sigma is None:
        x = np.load('data.npz')['x']
        y = np.load('data.npz')['y']
        sigma = np.load('data.npz')['sigma']

    fig, ax = plt.subplots()

    # Plot ground truth curve
    ax.plot(x, y, 'r', label='Ground Truth')
    ax.fill_between(x, y - sigma, y + sigma, color='r', alpha=0.2)

    # Plot predictions if available
    if predictions is not None:
        ax.plot(x, predictions, 'b', label='Prediction')

    # Load noisy data points and plot
    t = np.load('data.npz')['t']
    ax.scatter(x, t, label='Noisy Data', color='g')

    # Set title with MAP/ML, polynomial degree, and N
    ax.set_title(f"{model_name} Results (Degree {degree}, N={N})")

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.legend()

    plt.savefig(f'results/{model_name}_Degree_{degree}_N_{N}.png')
    plt.close(fig)

def calculate_rmse(predictions, true_values):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and true values.
    Args:
        predictions: Predicted target values.
        true_values: True target values.
    Returns:
        RMSE value.
    """
    return np.sqrt(np.mean((predictions - true_values) ** 2))


# TODO: Use the existing functions to create/load data as needed. You will now call your linear regression model
# to fit the data and plot the results.
def linear_regression(N=50):
    """
    Fits the ML and MAP models to the data, predicts values, and visualizes results
    for varying polynomial degrees.
    Args:
        N: Number of data points used for the experiment.
    """
    # Load data
    data = np.load('data.npz')
    x = data['x']
    y = data['y']
    t = data['t']

    # Degrees of the polynomial to evaluate
    degrees = [3, 6, 9]

    # Initialize ML and MAP models
    ml_model = ML()
    map_model = MAP(alpha=0.005, beta=11.1)

    # Loop over each degree
    for degree in degrees:
        print(f"Fitting models for polynomial degree {degree} with N={N}")

        # Fit the models
        ml_model.fit(x, t, degree=degree)
        map_model.fit(x, t, degree=degree)

        # Predict using the models
        ml_predictions = ml_model.predict(x, degree=degree)
        map_predictions = map_model.predict(x, degree=degree)

        # Calculate RMSE
        ml_rmse = calculate_rmse(ml_predictions, t)
        map_rmse = calculate_rmse(map_predictions, t)
        print(f"Degree {degree}, N={N} -> ML RMSE: {ml_rmse}, MAP RMSE: {map_rmse}")

        # Plot results for ML
        plot_with_shadded_bar(x, y, data['sigma'], predictions=ml_predictions, model_name="ML", degree=degree, N=N)

        # Plot results for MAP
        plot_with_shadded_bar(x, y, data['sigma'], predictions=map_predictions, model_name="MAP", degree=degree, N=N)

def main():
    """
    Takes N as a command-line argument, generates noisy data, and runs the regression experiment.
    """
    parser = argparse.ArgumentParser(description="Run polynomial regression experiments with ML and MAP models.")
    parser.add_argument("--N", type=int, default=50, help="Number of data points to generate")
    args = parser.parse_args()

    # Generate data and run the regression
    generateNoisyData(args.N)
    plot_with_shadded_bar(N=args.N)
    linear_regression(N=args.N)


if __name__ == '__main__':
    main()
