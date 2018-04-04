import pandas as pd
import numpy as np

class isolation_tree:
    
    def __init__(self, threshold=0., random_state=None):
        """
        The concept of an isolation tree is that outlier
        points should be easier to segment from the population
        at large. To exploit this, random segmentation of the
        data is done using a branching system where each node
        is defined by a randomly selected column and a randomly
        selected value. Points with values larger than that value
        go to one branch, the rest to the other. Each point is 
        measured defined by the number of cuts required to isolate
        the point. If it takes less than some threshold, that point 
        is considered an outlier for that tree. Because of the 
        random nature of this, many trees must be ensembled to 
        truly get an understanding of the outlier-ness. So
        these trees should only be used in groups. 
        See Isolation Forest.
        ---
        KWargs:
        threshold: how many standard deviation percentages below
        the mean should we use as the cutoff for outlierness
        random_state: sets the random seed for reproducibility
        """
        self.tree = self.tree_split(depth=0)
        self.threshold = threshold
        self.mean_depth = None
        self.std_depth = None
        self.depth_tracker = []
        if random_state is not None:
            np.random.seed(random_state)
    
    # Sub class for handling recursive nodes (only makes sense in the scope of a tree)
    class tree_split:
        """
        A sub class for handling recursive nodes. Each node will contain the value and column
        for the current split, as well as links to the resulting nodes from the split. The 
        results attribute remains empty unless the current node is a leaf. 
        """
        def __init__(self,col=-1,value=None,results=None,label=None,tb=None,fb=None,depth=None):
            self.col=col # column index of criteria being tested
            self.value=value # vlaue necessary to get a true result
            self.results=results # dict of results for a branch, None for everything except endpoints
            self.tb=tb # true decision nodes 
            self.fb=fb # false decision nodes
            self.depth=depth
    
    def split_data(self, X, colnum, value):
        """
        Returns: Two sets of data from the initial data. Set 1 contains those that passed
        the condition of data[colnum] >= value
        ----------
        Input: The dataset, the column to split on, the value on which to split
        """
        splitter = None
        if isinstance(value, int) or isinstance(value,float):
            splitter = lambda x: x[colnum] >= value
        else:
            splitter = lambda x: x[colnum] == value
        split1 = [i for i,row in enumerate(X) if splitter(row)]
        split2 = [i for i,row in enumerate(X) if not splitter(row)]
        set1X = X[split1]
        set2X = X[split2]
        return set1X, set2X

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
    
    def fit(self, X):
        """
        Helper function to wrap the fit method. This makes sure the full nested, 
        recursively built tree gets assigned to the correct variable name and 
        persists after training.
        """
        self.tree = self._fit(X)
        self.mean_depth = np.mean(self.depth_tracker)
        self.std_depth = np.std(self.depth_tracker)
        
    
    def _fit(self, X, depth=0):
        """
        Builds the isolation tree by recursively choosing 
        a random column and value, then splitting the data.
        If this results in a single point being isolated,
        the depth of this point is stored and an end node is
        created. Otherwise, the split datasets are segmented
        with another random column and value. This repeats
        until every point has been isolated and a nested
        tree structure is built of all the nodes. 
        ---
        Input: X (feature matrix)
        Output: A nested tree built upon the node class."""
        X = self.convert_to_array(X)

        if len(X) == 0: return tree_split()
        self.data_cols = X.shape[1]
        
        
        # Here we go choose a random feature, then a random value between the 
        # min and max values of that feature from the training data.
        
        col = np.random.randint(self.data_cols)
        value = np.random.uniform(min(X.T[col]), max(X.T[col]))
        set1, set2 = self.split_data(X, col, value)       
        
        # Now decide whether it's an endpoint or we need to split again.
        
        if len(X) > 1:
            true_branch = self._fit(set1, depth=depth+1)
            false_branch = self._fit(set2, depth=depth+1)
            return self.tree_split(col=col, value=value,
                    tb=true_branch, fb=false_branch, depth=depth)
        else:
            self.depth_tracker.append(depth)
            return self.tree_split(results=X, depth=depth)
       
    def print_tree(self, indent="---"):
        """
        Helper function to make sure the correct tree gets printed.
        ---
        In: indent (how to show splits between nodes)
        """
        self.__original_indent = indent
        self._print_tree_(self.tree, indent)
    
    def _print_tree_(self, tree, indent):
        """
        Goes through node by node and reports the column and value used to split
        at that node. All sub-nodes are drawn in sequence below the node.
        """
        if tree.results is not None: # if this is a end node
            print(str(tree.results))
        else:
            print('Column ' + str(tree.col)+' : '+str(tree.value)+'? ')
            # Print the branches
            print(indent+' True: ', end=' ')
            next_indent = indent+self.__original_indent
            self._print_tree_(tree.tb,indent=next_indent)
            print(indent+' False: ', end=' ')
            self._print_tree_(tree.fb,indent=next_indent)

    def predict(self, newdata):
        """
        Helper function to make sure the correct tree is used to
        make predictions. Also manages multiple rows of input data
        since the tree must predict one at a time.
        ---
        In: new data point of the same structure as the training X.
        Out: numpy array of the resulting predictions
        """
        results = []
        for x in newdata:
            results.append(self._predict(x,self.tree))
        results = self.convert_to_outliers(np.array(results))
        return results
            
    def _predict(self, newdata, tree):
        """
        Uses the recursive structure of the tree to follow each split for
        a new data point. If the node is an endpoint, return the depth
        of each point..
        """
        newdata = self.pandas_to_numpy(newdata)
        if tree.results is not None: # if this is a end node
            return tree.depth

        if isinstance(newdata[tree.col], int) or isinstance(newdata[tree.col],float):
            if newdata[tree.col] >= tree.value:
                return self._predict(newdata, tree.tb)

            else:
                return self._predict(newdata, tree.fb)
        else:
            if newdata[tree.col] == tree.value:
                return self._predict(newdata, tree.tb)
            else:
                return self._predict(newdata, tree.fb) 
            
    def convert_to_outliers(self, results):
        """
        Given an array of the "depth" of each point in the
        tree, decide whether it is a short enough path to
        be considered an outlier within the tree. 
        ---
        Inputs: array of tree depths for each point
        """        
        find_outliers_filter = results < (self.mean_depth - self.threshold*self.std_depth)
        results[find_outliers_filter] = -1.
        results[~find_outliers_filter] = 1.
        return results

class isolation_forest:
    
    def __init__(self, n_trees = 10, threshold=0., random_state=None):
        """
        The concept of an isolation tree is that outlier
        points should be easier to segment from the population
        at large. To exploit this, random segmentation of the
        data is done using a branching system where each node
        is defined by a randomly selected column and a randomly
        selected value. Points with values larger than that value
        go to one branch, the rest to the other. Each point is 
        measured defined by the number of cuts required to isolate
        the point. If it takes less than some threshold, that point 
        is considered an outlier for that tree. Because of the 
        random nature of this, many trees must be ensembled to 
        truly get an understanding of the outlier-ness. So
        these trees should only be used in groups. Once all the 
        trees are built, the each tree votes on whether each point
        is an outlier and the wisdom of all tress is reported.
        ---
        KWargs:
        n_trees: how many isolation trees to use in ensemble
        threshold: how many standard deviation percentages below
        the mean should we use as the cutoff for outlierness
        random_state: sets the random seed for reproducibility
        """
        self.n_trees = n_trees
        self.threshold = threshold
        self.trees = []
        self.random_state = random_state
    
    def fit(self, X):
        """
        Builds n_trees individual isolation trees and stores
        the trees for later usage in prediction.
        """
        X = self.convert_to_array(X)
       
        for ix in range(self.n_trees):
            if self.random_state is not None:
                new_tree = isolation_tree(self.threshold, random_state=self.random_state+ix)
            else:
                new_tree = isolation_tree(self.threshold)
            new_tree.fit(X)
            self.trees.append(new_tree)
    
    def predict(self, X):
        """
        Uses the list of tree models built in the fit, doing a predict with each
        model. This is handled by the decision_function method. All points that
        are seen as outliers in the majority of trees are marked as -1. Otherwise
        the value is 1. This -1, 1 choice is returned.
        ---
        Input: X (array, dataframe, or series)
        Output: Class ID (int)
        """
        X = self.convert_to_array(X)
        
        ensemble_predict = self.decision_function(X)
        ensemble_predict[ensemble_predict < 0] = -1.
        ensemble_predict[ensemble_predict >= 0] = 1.
        return ensemble_predict
    
    def decision_function(self, X):
        """
        For each tree built in the fit, make a prediction of
        outlier or not (-1 or 1). Then average over all the
        predictions to create a "strength of outlier" measure.
        -1 is every tree thinks this point is an outlier. 1 is
        every tree thinks this point is an inlier. 0 is an equal
        number of trees think in vs out. This "strenght" indicator
        can be used as a measure for how sure the ensemble is that
        a point is an outlier.
        """
        self.predicts = []
        for tree in self.trees:
            predictions = tree.predict(X).reshape(-1,1)
            self.predicts.append(predictions)
        
        all_tree_predictions = np.hstack(self.predicts)
        ensemble_predict = np.mean(all_tree_predictions, axis=1)
        return ensemble_predict
    
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