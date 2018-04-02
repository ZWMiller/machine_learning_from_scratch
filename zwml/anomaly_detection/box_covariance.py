import numpy as np
import pandas as pd

class box_covariance:
    
    def __init__(self, threshold=1.): 
        """
        Builds a box envelope around the data using a
        standard deviation threshold. Any points within this
        box are considered inliers, and points outside of this
        box are considered outliers. This is a fairly simplistic
        method that is not very robust to highly correlated
        data with "close by" outliers.
        ---
        KWargs:
        threhsold: how many standard deviations do you want
        to consider an "inlier"
        """
        self.threshold = threshold
        self.data_stats = {}
        self.number_of_columns = None
        
    def fit(self, X):
        """
        Learns about the input data and stores the mean and 
        standard deviation of each column.
        ---
        In: X (features); np.array or pandas dataframe/series
        """
        X = self.convert_to_array(X)
        self.number_of_columns = X.shape[1]
        
        for ix in range(self.number_of_columns):
            col = X.T[ix]
            col_mean = np.mean(col)
            col_std = np.std(col)
            self.data_stats[ix] = (col_mean, col_std)
            
    def predict(self, X):
        """
        For each data point, subtract the mean of the column
        and then see if the data point is within 
        threshold*std_dev of that column of 0. If so, it's an
        inlier. Otherwise it's an outlier.
        """
        X = self.convert_to_array(X)
        result = np.ones(X.shape[0])
        for ix in range(self.number_of_columns):
            X.T[ix] = X.T[ix] - self.data_stats[ix][0]
            result[(result != -1) & (np.abs(X.T[ix]) >= self.data_stats[ix][1]*self.threshold)] = -1
        return result
    
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