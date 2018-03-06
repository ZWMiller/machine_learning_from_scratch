import numpy as np
from copy import copy


class normalizer:
    
    def __init__(self, axis='col'):
        """
        Normalizer has two behaviors. If the axis is 'col', it divides
        each column by the maximum magnitude in that column. If the axis 
        is 'row', it forces each row to sum to 1.
        ---
        KWargs:
        axis: mode of behavior. See description for details.
        """
        self.axis = axis
        self.data_stats = {}
        self.number_of_columns = None
        if self.axis not in ['col', 'row']:
            raise ValueError("axis must be either 'row' or 'col'")
        
    def fit(self, X):
        """
        If axis='col', learns about the input data and 
        stores the max value of each column. If set for 
        'row', does nothing.
        ---
        In: X (features); np.array or pandas dataframe/series
        """
        X = self.convert_to_array(X)
        self.number_of_columns = X.shape[1]
        
        if self.axis == 'col':
            for ix in range(self.number_of_columns):
                self.data_stats[ix] = np.amax(np.abs(X.T[ix]))
    
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
        
        if self.axis == 'col':
            for ix in range(self.number_of_columns):
                new_X.T[ix] = new_X.T[ix]/self.data_stats[ix]
            
        if self.axis == 'row':
            new_X = new_X/np.sum(new_X**2, axis=1).reshape(-1,1)
            
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