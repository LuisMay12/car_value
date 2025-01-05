"""
    Same as model.py but with optimized code for the functions compute_cost and compute_gradient.
    You can notice that the code is more concise and efficient. Also way more faster.
"""

import numpy as np
import copy, math


"""
    Args:
      x (ndarray (m, n)): Training data with m examples and n features.
      y (ndarray (m,)): Target values corresponding to the input data.
      w (ndarray (n,)): Weights of the linear model.
      b (scalar): Bias term of the linear model.
      
    Returns:
      total_cost (float): Computed cost based on the mean squared error formula.
      
    Description:
    Computes the cost function for linear regression:
      J(w,b) = (1/2m) * sum((f_{w,b}(x) - y)^2),
    where f_{w,b}(x) = np.dot(w, x) + b.
"""
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    total_cost = 0.0
    
    # Calculate cost with vectorized computation
    f_wb = np.dot(x, w) + b
    
    # remember here that the mean is the sum of the squared differences divided by
    # the num of values (m)
    total_cost = np.mean((f_wb - y)**2) / 2

    return total_cost

"""
    Computes the gradient of the cost function with respect to weights and bias.
    
    Args:
      X (ndarray (m, n)): Data, m examples with n features.
      y (ndarray (m,)): Target values.
      w (ndarray (n,)): Model weights.
      b (scalar): Model bias.
      
    Returns:
      dj_dw (ndarray (n,)): Gradient of the cost with respect to weights.
      dj_db (scalar): Gradient of the cost with respect to bias.
"""
def compute_gradient(X, y, w, b): 

    m = X.shape[0]           #(number of examples, number of features)

    f_wb = np.dot(X, w) + b
    err = f_wb - y
    dj_dw = np.dot(X.T, err) / m
    dj_db = np.sum(err) / m
        
    return dj_db, dj_dw

"""
    Performs batch gradient descent to optimize weights and bias.
    
    Args:
      X (ndarray (m, n)): Training data with m examples and n features.
      y (ndarray (m,)): Target values.
      w_in (ndarray (n,)): Initial weights.
      b_in (scalar): Initial bias.
      cost_function (function): Function to compute the cost.
      gradient_function (function): Function to compute the gradients.
      alpha (float): Learning rate.
      num_iters (int): Number of iterations to run gradient descent.
      
    Returns:
      w (ndarray (n,)): Optimized weights.
      b (scalar): Optimized bias.
      J_history (list): History of cost values over iterations.
"""
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    # An array to store cost J and w's at each iteration
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b) 

        # Update Parameters using w, b, alpha and gradient
        w -= alpha * dj_dw
        b -= alpha * dj_db
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

"""
    Normalizes the features of the dataset.
    
    Args:
      X (ndarray (m, n)): Dataset with m examples and n features.
      
    Returns:
      X_normalized (ndarray (m, n)): Normalized dataset.
      mean (ndarray (n,)): Mean of each feature.
      std (ndarray (n,)): Standard deviation of each feature.
"""
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

# Simulate dataset for linear regression
"""
    Generates a synthetic dataset for training a linear regression model.
    The dataset simulates the market value of cars based on kilometers driven 
    and years of manufacture.
"""
def main_data():
    np.random.seed(42)
    n_samples = 500
    kilometers = np.random.randint(0,200001, n_samples).reshape(-1,1)
    years = np.random.randint(2000,2024, n_samples).reshape(-1,1)
    base_value = 30000
    coef_km = -0.05
    coef_year = -1000
    noise = np.random.normal(0, 2000, n_samples).reshape(-1,1)
    market_value = base_value + coef_km * kilometers + coef_year * (2025 - years) + noise

    # train data
    """
        Prepares the feature matrix and target vector for training.
    """
    X = np.hstack([np.ones((n_samples, 1)), kilometers, 2025 - years])
    y = market_value.reshape(-1,)

    # Normalize features (except the bias term)
    X_normalized, mean_X, std_X = normalize_features(X[:, 1:])


    """
        Initializes parameters and runs gradient descent to find optimal weights and bias.
    """
    b_init = 30000.0
    w_init = np.zeros(X_normalized.shape[1])
    alpha = 0.01
    iterations = 2000
    w_final, b_final, J_history = gradient_descent(X_normalized, y, w_init, b_init, compute_cost, compute_gradient, alpha, iterations)

    print(f"Final model parameters: w = {w_final}, b = {b_final}")
    return w_final, b_final,  mean_X, std_X