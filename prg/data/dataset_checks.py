import numpy as np

def check_dataset(X):
    X = check_dataset_shape(X)
    #check_variable_type_in_dataset(X)
    return X

def check_variable_type_in_dataset(X):
    """
    This function checks whether the dataset contains binary variables and what these binary values are.
    """
    X_unique = np.unique(X)

    if len(X_unique) == 2:
        print(f"- Dataset contains binary values [{X_unique[0]}, {X_unique[1]}].")
    elif len(X_unique) == 3:
        print(f"- Dataset contains triple values [{X_unique[0]}, {X_unique[1]}, {X_unique[2]}]")
    else:
        print(f"- Dataset does not contain binary values. Found {len(X_unique)} unique values.")

def check_dataset_shape(X):
    """
    Computation of the correlation matrix assumes the shape = (n_features, n_datapoints). 
    Perform a simple check and ask user if they want to transpose the data.
    """
    if len(X[:, 0]) > len(X[0]):
        response = input(f"Dataset has shape = {X.shape}. There are more features than data points! Do you want to transpose the data? ")
        if response in ["yes", "y", "YES", "ye", "yh", "Yes"]:
            return X.T
    return X
