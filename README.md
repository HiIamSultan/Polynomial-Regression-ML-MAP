# Polynomial-Regression-ML-MAP

This repository contains the code for **Polynomial Regression** using **Maximum Likelihood (ML)** and **Maximum A Posteriori (MAP)** methods. The project evaluates these models under varying polynomial degrees ( M ) and dataset sizes ( N ).

---

## Repository Structure

This folder contains the following files and folder:

- `results`: Contains saved plots for ML and MAP predictions for each degree ( M ) and dataset size ( N ).

- `data.npz`:
  - Contains:
    - `x`: Input data points.
    - `y`: Ground truth sine function values.
    - `t`: Noisy data points.
    - `sigma`: Standard deviation of the noise.

- `linear_regression.py`:

  - `generateNoisyData`: Generates noisy sample points from a sine function and saves the dataset in a `.npz` file.
  - `calculate_rmse`: calculates RMSE.
  - `plot_with_shadded_bar`: A visualization function to draw curve with shaded error bar.
  - `linear_regression`: Fits ML and MAP models to the dataset for varying degrees ( M = 3, 6, 9 ). It contains a sample script to load data, plot points and curves.
  - `main`: The entry point for running the project pipeline. Takes the dataset size  N  as a command-line argument and runs the entire process.

- `model.py`:

  - `ML`:
    - Implements Maximum Likelihood estimation.
    - Includes `fit` (to calculate weights) and `predict` (to predict target values).
  - `MAP`:
    - Implements Maximum A Posteriori estimation with regularization.
    - Includes `fit` (to calculate regularized weights) and `predict` (to predict target values).

---

### **Packages**

The project uses the following packages:

- `numpy`
- `matplotlib`
- `argparse`

To ensure compatibility, you are recommended to use a virtual environment.

## Setting-up Your Python Environment
0.	Make sure you have Python installed. This code is compatible with `Python 3.13` but you can try the older versions as well. 
1.	Create a virtual environment (optional but recommended) to isolate your project dependencies. 
	For example, if you're using `conda` as your python manager, run 
		
			conda create --name PRML_env python=3.13
	Activate the environment by running
			
			conda activate PRML_env
	To delete the environment, use the following command:
	
			conda env remove -n PRML_env
2.	When you're done working with the project, you can deactivate the virtual environment using:

			conda deactivate

### **Installation**

1. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
2. Clone the repository and navigate to the project directory.

---

## How to Run

Run the `linear_regression.py` script with a specified dataset size  N :

```bash
python linear_regression.py --N <dataset_size>
```

Examples:

```bash
python linear_regression.py --N 50
python linear_regression.py --N 1000
```

---

## Expected Outputs

1. **Console Output**:

   - RMSE values for ML and MAP models for degrees  M = 3, 6, 9  and dataset sizes ( N = 50, 1000 ).
   - Example:
     ```
     Fitting models for polynomial degree 3 with N=50
     Degree 3, N=50 -> ML RMSE: 0.2825, MAP RMSE: 0.2825
     ```

2. **Plots**:

   - Generated plots are saved in the `results/` directory.
   - Each plot includes:
     - **Ground Truth (red curve)**: The sine function.
     - **Noisy Data (green dots)**: Sample points with added noise.
     - **Predictions (blue curve)**: ML and MAP predictions for each degree.

---

## Observations

- **Polynomial Degree ( M ):**

  - Lower degrees ( M = 3 ) result in underfitting, while higher degrees ( M = 9 ) may overfit small datasets.
  - MAP mitigates overfitting at higher degrees due to regularization.

- **Dataset Size ( N ):**

  - Smaller datasets ( N = 50 ) amplify the noise, making MAP more effective in controlling overfitting.
  - Larger datasets ( N = 1000 ) reduce noise influence, and both ML and MAP converge in performance.

- **Central Limit Theorem Validation:**

  - Larger datasets produce smoother predictions and reduced variability, confirming the CLT.

---

If you face any issues while running the repository, please email mqm7099@psu.edu

