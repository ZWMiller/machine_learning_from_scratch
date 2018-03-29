import numpy as np
import pandas as pd
from itertools import combinations

class agglomerative_clustering:
    
    def __init__(self, linkage="average", n_clusters=5, max_dist=None):
        """
        Agglomerative clustering uses a "linkage" function to measure
        how close together two current clusters are. It then merges
        the two closest clusters into a single bigger cluster. This
        process is repeated until there are n_clusters remaining,
        or some other cut-off is applied. If no cut-off applied,
        will eventually result in a single cluster of all data points.
        ---
        KWargs: 
        linkage: how to measure cluster closeness. Options 
        ('average','complete','minimal','ward')
        n_clusters: when n_clusters is reached, stop merging
        max_dist: if no clusters are closer than max_dist, stop merging
        """
        self.link = linkage
        self.clusters = {}
        self.n_clusters = n_clusters
        self.max_dist = max_dist
        self.merge_tracker = []
        self.data = None
        self.labels = None

    def euclidean_distance(self, pt1, pt2):
        """
        Returns the distance. Currently only uses Euclidean distance.
        ---
        Input: Cluster (cluster object), data point (np array)
        Output: Distance (float)
        """
        return np.sqrt(np.sum((pt1 - pt2)**2))
       
    def compute_distance(self, idx1, idx2):
        """
        Chooses how do decide "how close" two clusters are. Applies to
        proper measure and returns it.
        """
        if self.link == 'average':
            return self.average_linkage(idx1, idx2)
        elif self.link == 'complete':
            return self.complete_linkage(idx1, idx2)
        elif self.link == 'minimal':
            return self.minimal_linkage(idx1, idx2)
        elif self.link == 'ward':
            return self.ward_linkage(idx1, idx2)
        else:
            raise TypeError("Not a proper linkage function selection!")
        
    def average_linkage(self, idx1, idx2):
        """
        Finds the distance between the mean of cluster 1 and the mean
        of cluster 2.
        """
        return self.euclidean_distance(self.clusters[idx1]['mean'], self.clusters[idx2]['mean'])
    
    def complete_linkage(self, idx1, idx2):
        """
        Finds the maximum possible distance between points in 
        cluster 1 and cluster 2. Meaning it returns the distance of the
        two points in the clusters that are furthest apart.
        """
        max_dist = 0.
        for pt in self.clusters[idx1]['members']:
            for pt2 in self.clusters[idx2]['members']:
                dist = self.euclidean_distance(self.data[pt], self.data[pt2])
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def minimal_linkage(self, idx1, idx2):
        """
        Finds the minimum possible distance between points in 
        cluster 1 and cluster 2. Meaning it returns the distance of the
        two points in the clusters that are nearest together.
        """
        min_dist = 99999999.
        for pt in self.clusters[idx1]['members']:
            for pt2 in self.clusters[idx2]['members']:
                dist = self.euclidean_distance(self.data[pt], self.data[pt2])
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    
    def ward_linkage(self, idx1, idx2):
        """
        Measures how far every point in each cluster is from its own
        cluster mean, called the inertia. Then "pretends to merge" the
        points and measures the inertia of the resulting mega-cluster.
        Returns the "gained" inertia by the pretend merge. 
        """
        inertia_1 = 0
        inertia_2 = 0
        inertia_combined = 0
        
        for pt in self.clusters[idx1]['members']:
            inertia_1 += self.euclidean_distance(self.data[pt], self.clusters[idx1]['mean'])
        for pt in self.clusters[idx2]['members']:
            inertia_2 += self.euclidean_distance(self.data[pt], self.clusters[idx2]['mean'])
            
        combined_members = self.clusters[idx1]['members'] + self.clusters[idx2]['members']
        combined_mean = np.mean([X[i] for i in combined_members], axis=0)
        
        for pt in combined_members:
            inertia_combined += self.euclidean_distance(self.data[pt], combined_mean)
            
        return inertia_combined - inertia_1 - inertia_2
        
    def init_clusters(self, X):
        """
        Create a lookup table where each point is its own cluster.
        As we merge clusters, we'll remove members and track the progress
        with this dictionary.
        """
        for idx, pt in enumerate(X):
            self.clusters[idx] = {'members': [idx], 'mean': pt}
        self.data = X
            
    def merge_clusters(self, idx1, idx2, distance):
        """
        Takes two clusters and makes them into a single, 
        larger cluster. Also tracks the "distance" that the merge
        occurred at for future reference.
        """
        self.clusters[idx1]['members'] += self.clusters[idx2]['members']
        self.clusters[idx1]['mean'] = np.mean([X[i] for i in self.clusters[idx1]['members']], axis=0)
        self.clusters.pop(idx2, None)
        self.merge_tracker.append((idx1, idx2, distance))
    
    def fit(self, X):
        """
        Makes ever point into it's own cluster. Checks the 
        linkage distance for all possible merges (using the 
        combinations to see what merges are possible). Whatever 
        clusters have the smallest linkage relationship are merged 
        together into a new cluster which takes the id of the lower
        numbered cluster. Tracks the "size" of each merge for
        review. Repeat this until down to n_clusters or the distance
        is larger than the allowed maximum. Then label the clusters.
        ---
        Input: X (data, array/dataframe)
        """
        X = self.convert_to_array(X)
        self.init_clusters(X)
        
        while len(self.clusters.keys()) > self.n_clusters:
            decision_tracker = {}
            for combo in combinations(self.clusters.keys(), r=2):
                decision_tracker[combo] = self.compute_distance(combo[0], combo[1])
            to_merge = sorted(decision_tracker.items(), key=lambda x: x[1])[0][0]
            
            if self.max_dist != None and self.linkage != 'ward' and decision_tracker[combo] > self.max_dist:
                break
                
            self.merge_clusters(to_merge[0], to_merge[1], decision_tracker[combo])
        
        self.labels = np.zeros(X.shape[0])
        for ix, clst in enumerate(self.clusters.keys()):
            members = self.clusters[clst]['members']
            self.labels[members] = ix
    
    def fit_predict(self,X):
        """
        Creates clusters for data X, and returns cluster ID's for each point.
        ---
        Input: X (data, array)
        Output: cluster IDs for X (array)
        """
        self.fit(X)
        return self.labels
    
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
    