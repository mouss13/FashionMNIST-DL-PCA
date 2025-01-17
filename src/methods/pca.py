import numpy as np
import matplotlib.pyplot as plt

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None
        self.explained_variance_ratio_ = None
    
    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.mean = np.mean(training_data, axis=0)
        X_tilde = training_data - self.mean
        C = np.cov(X_tilde.T)

        eigvals, eigvecs = np.linalg.eigh(C)

        sorted_idx = np.argsort(eigvals[::-1])
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        self.W = eigvecs[:, :self.d]

        exvar = (np.sum(eigvals[:self.d]) / np.sum(eigvals)) * 100
        
        # calculate and store the explained variance ratio
        self.explained_variance_ratio_ = eigvals[:self.d] / np.sum(eigvals)

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        centered_data = data - self.mean
        data_reduced = np.dot(centered_data, self.W)
        return data_reduced
        
