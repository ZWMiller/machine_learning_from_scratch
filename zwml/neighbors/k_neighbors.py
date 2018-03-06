import numpy as np
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
        self.X = copy.copy(self.convert_to_array(X))
    
    def find_neighbors(self, X):
        """
        Iterates through all points to predict, calculating the distance
        to all of the training points. It then finds the closest points.
        ___
        In: new data to predict (np.array, pandas series/dataframe)
        Out: predictions (np.array)
        """
        X = self.convert_to_array(X)
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
                #results.append([x[1] for x in neighbors])
        return np.array(results)

    def dist_between_points(self, a, b):
        """
        Calculates the distance between two vectors.
        ---
        Inputs: a,b (np.arrays)
        Outputs: distance (float)"""
        assert np.array(a).shape == np.array(b).shape, 'Vectors must be of same size'
        return np.sqrt(np.sum((a-b)**2))
    
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