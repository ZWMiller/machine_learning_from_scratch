import math
import numpy as np
import pandas as pd
import copy
import collections

class KNearestNeighbors:
    
    def __init__(self, n_neighbors=5):
        """
        KNearestNeighbors is a distance based classifier that returns
        predictions based on the nearest points in the feature space.
        ---
        In: n_neighbors (int) - how many closest neighbors do we consider
        """
        self.k = int(n_neighbors)
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
        self.X = copy.copy(self.pandas_to_numpy(X))
        self.y = copy.copy(self.pandas_to_numpy(y))
        
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
        X = self.pandas_to_numpy(X)
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
        returning the mode vote for the n_neighbors (k) closest votes. 
        ---
        In: [[distance, label]] list of lists
        Output: class label (int)
        """
        results = sorted(results, key=lambda x: x[0])
        dists, votes = zip(*results)
        return collections.Counter(votes[:self.k]).most_common(1)[0][0]

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
        Uses the predict method to measure the accuracy of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: accuracy (float)
        """
        pred = self.predict(X)
        correct = 0
        for i,j in zip(y,pred):
            if i == j:
                correct+=1
        return float(correct)/float(len(y))
