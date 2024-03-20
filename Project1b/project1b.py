# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))

    # TODO: Enter your code here
    X_linear_features = X
    X_quadratic_features = np.square(X)
    X_exponential_features = np.exp(X)
    X_cosine_features = np.cos(X)
    X_constant_feature = np.ones((700,1))

    X_transformed = np.concatenate((X_linear_features, X_quadratic_features, X_exponential_features, X_cosine_features, X_constant_feature), axis=1)
    
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transforms them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # TODO: Enter your code here

    # Calculate Squared Loss calculate this both with using the max. norm and the mean error approach 
    #and see what performs better. I will to the mean error fist
    def compute_loss(w, X, y):
        prediction = X @ w
        loss = np.sqrt(np.mean((prediction - y) ** 2))
        return loss
    
    # Compute the gradient of the loss with respect to w
    def compute_gradient(w, X, y):
        prediction = X @ w
        gradient = 2 * X.T @ (prediction - y) / X.shape[0]
        return gradient

    # Gradient descent algorithm

    learning_rate = 0.01  # This is a hyperparameter you'll need to choose
    max_iter = 1000  # Another hyperparameter
    for i in range(max_iter):
        # Compute the loss
        loss = compute_loss(w, X_transformed, y)

        # Compute the gradient
        gradient = compute_gradient(w, X_transformed, y)

        # Update the weights
        w -= learning_rate * gradient

        # Print the loss every 100 iterations (or any number of your choice)
        if i % 100 == 0 or i == 999:
            print(f"Iteration {i}, Loss: {loss}")


    assert w.shape == (21,)
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
