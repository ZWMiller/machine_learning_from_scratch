import numpy as np
import pandas as pd
import copy

class k_neighbors:
    
    def __init__(self, n_neighbors=5, return_dist=False):
        """
        KNearestNeighbors finds the nearest points in the feature space.
        ---
        In: n_neighbors (int) - how many closest neighbors do we consider
        """
        if n_neighbors > 0:
            self.k = int(n_neighbors)
        else:
            print("n_neighbors must be >0. Set to 5!")
            self.k = 5
        self.X = None
        self._return_dist = return_dist
        
    def fit(self, X):
        """
        Makes a copy of the training data that can live within the class.
        Thus, the model can be serialized and used away from the original
        training data. 
        ---
        In: X (features); np.array or pandas dataframe/series
        """
        self.X = copy.copy(self.pandas_to_numpy(X))
        
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
    
    def find_neighbors(self, X):
        """
        Iterates through all points to predict, calculating the distance
        to all of the training points. It then finds the closest points.
        ___
        In: new data to predict (np.array, pandas series/dataframe)
        Out: predictions (np.array)
        """
        X = self.pandas_to_numpy(X)
        results = []
        for x in X:
            local_results = []
            for x2 in self.X:
                local_results.append([self.dist_between_points(x,x2),x2])
            neighbors = sorted(local_results, key=lambda x: x[0])[:self.k]
            if self._return_dist:
                results.append(neighbors)
            else:
                for x in neighbors:
                    results.append(x[1])
        return np.array(results)

    def dist_between_points(self, a, b):
        """
        Calculates the distance between two vectors.
        ---
        Inputs: a,b (np.arrays)
        Outputs: distance (float)"""
        assert np.array(a).shape == np.array(b).shape, 'Vectors must be of same size'
        return np.sqrt(np.sum((a-b)**2))