import numpy as np
import pandas as pd

class dbscan:
    
    def __init__(self, epsilon=0.5, min_points=5):
        self.epsilon = epsilon
        self.min_points = min_points
        self.data_cols = None
        self.labels_ = None
        self.neighbors = {}
        
    def fit(self, X):
        X = self.pandas_to_numpy(X)
        if not self.data_cols:
            self.data_cols = X.shape[1]
        self.check_feature_shape(X)
        self.visited_points = []
        self.clusters = []
        
        for ix in range(X.shape[0]):
            if ix in self.visited_points:
                continue
            self.neighbors[ix] = self.get_neighbors(ix, X)
            if len(self.neighbors[ix]) >= self.min_points:
                self.visited_points.append(ix)
                self.clusters.append(self.make_cluster(ix, X))
                
        self.labels_ = self.get_labels(X)
        
    def get_labels(self, X):
        labels = [-1]*X.shape[0]
        for clst_id, cluster in enumerate(self.clusters):
            for pt_id in cluster:
                labels[pt_id] = clst_id
        return np.array(labels)
            
    def make_cluster(self, ix, X):
        cluster = [ix]
        for neighbor in self.neighbors[ix]:
            if neighbor not in self.visited_points:
                self.visited_points.append(neighbor)
                self.neighbors[neighbor]= self.get_neighbors(ix, X)
                if len(self.neighbors[neighbor]) >= self.min_points:
                    cluster_from_neighbor = self.make_cluster(neighbor, X)
                    cluster = cluster + cluster_from_neighbor
                else:
                    cluster.append(neighbor)
        return cluster
          
    def fit_predict(self,X):
        self.fit(X)
        return self.labels_
    
    def get_neighbors(self, ix, X):
        neighbors = []
        pt = X[ix]
        for ix2, pt2 in enumerate(X):
            dist = np.sqrt(np.sum((pt2 - pt)**2)) 
            if dist <= self.epsilon:
                neighbors.append(ix2)
        return neighbors
        
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
    
   