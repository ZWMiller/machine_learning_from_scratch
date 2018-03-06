import math
import numpy as np
import copy
import collections

class knn_regressor:
    
    def __init__(self, n_neighbors=5):
        """
        KNearestNeighbors is a distance based regressors that returns
        predictions based on the nearest points in the feature space.
        ---
        In: n_neighbors (int) - how many closest neighbors do we consider
        """
        if n_neighbors > 0:
            self.k = int(n_neighbors)
        else:
            print("n_neighbors must be >0. Set to 5!")
            self.k = 5
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        """
        Makes a copy of the training data that can live within the class.
        Thus, the model can be serialized and used away from the original
        training data. 
        ---
        In: X (features), y (labels); both np.array or pandas dataframe/series
        """
        self.X = copy.copy(self.convert_to_array(X))
        self.y = copy.copy(self.convert_to_array(y))
        
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
    
    def predict(self, X):
        """
        Iterates through all points to predict, calculating the distance
        to all of the training points. It then passes that to a sorting function
        which returns the most common vote of the n_neighbors (k) closest training
        points.
        ___
        In: new data to predict (np.array, pandas series/dataframe)
        Out: predictions (np.array)
        """
        X = self.convert_to_array(X)
        results = []
        for x in X:
            local_results = []
            for (x2,y) in zip(self.X,self.y):
                local_results.append([self.dist_between_points(x,x2),y])
            results.append(self.get_final_predict(local_results))
        return np.array(results).reshape(-1,1)
            
    def get_final_predict(self,results):
        """
        Takes a list of [distance, label] pairs and sorts by distance,
        returning themean of the n_neighbors (k) closest points. 
        ---
        In: [[distance, label]] list of lists
        Output: class label (int)
        """
        results = sorted(results, key=lambda x: x[0])
        dists, votes = zip(*results)
        return np.mean(votes[:self.k])

    def dist_between_points(self, a, b):
        """
        Calculates the distance between two vectors.
        ---
        Inputs: a,b (np.arrays)
        Outputs: distance (float)"""
        assert np.array(a).shape == np.array(b).shape
        return np.sqrt(np.sum((a-b)**2))
    
    def score(self, X, y):
        """
        Uses the predict method to measure the (negative)
        mean squared error of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: negative mean squared error (float)
        """
        pred = self.predict(X)
        return -1.* np.mean((np.array(pred)-np.array(y))**2)