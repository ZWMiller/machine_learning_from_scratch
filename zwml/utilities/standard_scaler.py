import numpy as np
from copy import copy


class standard_scaler:
    
    def __init__(self, demean=True, dev_scale=True):
        """
        Standard Scaler demeans each column and converts 
        each column to have a standard deviation of 1.
        ---
        KWargs:
        demean: whether to subtract the mean from each column
        dev_scale: whether to convert to unit variance
        """
        self.demean = demean
        self.dev_scale = dev_scale
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
    
    def transform(self,X):
        """
        Given the information learned about the training data,
        remove the mean and scale the new data as requested by
        the user.
        ---
        In: X (features); np.array or pandas dataframe/series
        """
        X = self.convert_to_array(X)
        new_X = copy(X)
        
        for ix in range(self.number_of_columns):
            if self.demean:
                new_X.T[ix] = new_X.T[ix] - self.data_stats[ix][0]
            if self.dev_scale:
                new_X.T[ix] = new_X.T[ix]/self.data_stats[ix][1]
        
        return new_X
    
    def fit_transform(self, X):
        """
        Learn from X and then return the transformed version
        of X for the user to use.
        ---
        In: X (features); np.array or pandas dataframe/series
        """
        self.fit(X)
        return self.transform(X)
    
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