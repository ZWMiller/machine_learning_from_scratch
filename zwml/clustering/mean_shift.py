import numpy as np
import pandas as pd
from copy import copy

class mean_shift:
    
    def __init__(self, bandwidth=1, iters=10, threshold = .1):
        self._iters = iters
        self.bandwidth = bandwidth
        self.data_cols = None
        self.threshold = threshold
        
    def fit(self, X):
        X = self.pandas_to_numpy(X)
        if not self.data_cols:
            self.data_cols = X.shape[1]
        self.check_feature_shape(X)
        self._original_data = copy(X)
        
    def transform(self, X):
        X = self.pandas_to_numpy(X)
        if not self.data_cols:
            self.data_cols = X.shape[1]
        X = self.check_feature_shape(X)
        new_X = []
        for pt in X:
            movement = self.threshold+1
            it=0
            p = copy(pt)
            while it < self._iters and movement > self.threshold:
                shift = np.zeros(len(p))
                scale = 0.
                for orig_pt in self._original_data:
                    weight = self.rbf_kernel(p, orig_pt, sig=self.bandwidth)
                    shift += weight*orig_pt
                    scale += weight
                movement = p - shift/scale
                p = shift/scale
                movement = np.sqrt(np.sum(movement**2))
                it+=1
            new_X.append(p)
        return new_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def check_feature_shape(self, x):
        """
        Helper function to make sure any new data conforms to the fit data shape
        ---
        In: numpy array, (unknown shape)
        Out: numpy array, shape: (rows, self.data_cols)"""
        return x.reshape(-1,self.data_cols)
        
    def rbf_kernel(self, x1, x2, sig=1.):
        """
        Returns the rbf affinity between two points (x1 and x2),
        for a given bandwidth (standard deviation).
        ---
        Inputs: 
            x1; point 1(array)
            x2; point 2(array)
            sig; standard deviation (float)
        """
        diff = np.sum((x1-x2)**2)
        norm = 1/(np.sqrt(2*np.pi*sig**2))
        return norm*np.exp(-diff/(2*sig**2))
    
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