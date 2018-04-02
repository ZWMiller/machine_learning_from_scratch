import numpy as np
import pandas as pd
from copy import copy

class elliptic_covariance:
    
    def __init__(self, threshold=5.991): 
        """
        Uses the covariance matrix to find the eigenvalues
        and eigenvectors. Then finds an ellipse that represents
        the training data using the standard deviation. 
        The ellipse is based on the formula:
        (x/std_X)^2 + (y/std_y)^2 + (z/std_z)^2 + ... = threshold
        The threshold value will define the allowed inliers
        and their total deviation by "distance" from the mean.
        ---
        KWargs:
        threshold: how far from the mean do you want the inlier 
        surface to exist. 5.991 represents a 95% confidence interval
        from the Cumulative Chi_2 distribution.
        """
        self.threshold = threshold
        self.number_of_columns = None
        
    def fit(self, X):
        """
        Learns about the input data and stores the mean and 
        standard deviation of each column.
        ---
        In: X (features); np.array or pandas dataframe/series
        """
        X = self.convert_to_array(X)
        new_X = copy(X)
        self.number_of_columns = new_X.shape[1]
        
        self.means = np.mean(new_X, axis = 0)  
        new_X -= self.means
        cov = np.cov(new_X, rowvar = False)
        eigenvals , eigenvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigenvals)[::-1]
        self.eigenvecs = eigenvecs[:,idx]
        self.eigenvals = eigenvals[idx]

            
    def predict(self, X):
        """
        For each data point, compute whether each point
        lies within the ellipsoid created by
        (x/std_X)^2 + (y/std_y)^2 + (z/std_z)^2 + ... = threshold
        This is checked by converting each point to the new reduced
        eigen space, where the ellipsoid is centered on 0
        and each direction has an axis the size of the sqrt(eigenvalue)
        The standard deviation is that sqrt(eigenvalue) since the
        eigenvalue captures the variance in along the eigenvector.
        """
        X = self.convert_to_array(X)
        new_X = copy(X)
        new_X -= self.means
        new_X = self.convert_to_pca_space(new_X)  
        new_X /= np.sqrt(self.eigenvals)
        new_X = new_X**2
        result = np.ones(X.shape[0])
        result[np.sum(new_X, axis=1) >= self.threshold] = -1
        return result
    
    def convert_to_pca_space(self, X):
        """
        Converts the points to the new eigenspace
        """
        return np.dot(X,self.eigenvecs)  
    
    def fit_predict(self, X):
        """
        Learn from X and then return the transformed version
        of X for the user to use.
        ---
        In: X (features); np.array or pandas dataframe/series
        """
        self.fit(X)
        return self.predict(X)
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x) 
    
    def handle_1d_data(self,x):
        """
        Converts 1 dimensional data into a series of rows with 1 columns
        instead of 1 row with many columns.
        """
        if x.ndim == 1:
            x = x.reshape(-1,1)
        return x
    
    def convert_to_array(self, x):
        """
        Takes in an input and converts it to a numpy array
        and then checks if it needs to be reshaped for us
        to use it properly
        """
        x = self.pandas_to_numpy(x)
        x = self.handle_1d_data(x)
        return x