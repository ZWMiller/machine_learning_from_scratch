import sys 
sys.path.append('../..')
from zwml.tree_models import decision_tree_regressor
import collections
import pandas as pd
import numpy as np

class bagging_regressor:
    
    def __init__(self, n_trees = 10, max_depth=None):
        """
        Bagging regressor uses bootstrapping to generate n_trees different
        datasets and then applies a decision tree to each dataset. The final 
        prediction is an ensemble of all created trees.
        ---
        Params:
        n_trees (int): number of bootstrapped trees to grow for ensembling
        max_depth (int): maximum number of splits to make in each tree)
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def get_bagged_data(self, X, y):
        """
        Chooses random rows to populate a bootstrapped dataset, with replacement.
        Maintains the correlation between X and y
        ---
        Input: X, y (arrays)
        Outputs: randomized X,y (arrays)
        """
        index = np.random.choice(np.arange(len(X)),len(X))
        return X[index], y[index]
    
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
    
    def fit(self, X, y):
        """
        Generates the bootstrapped data then uses the decision tree
        class to build a model on each bootstrapped dataset. Each tree
        is stored as part of the model for later use.
        ---
        Input: X, y (arrays, dataframe, or series)
        """
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        for _ in range(self.n_trees):
            bagX, bagy = self.get_bagged_data(X,y)
            new_tree = decision_tree_regressor(max_depth=self.max_depth)
            new_tree.fit(bagX, bagy)
            self.trees.append(new_tree)
            
    def predict(self, X):
        """
        Uses the list of tree models built in the fit, doing a predict with each
        model. The final prediction uses the mode of all the trees predictions.
        ---
        Input: X (array, dataframe, or series)
        Output: Class ID (int)
        """
        X = self.pandas_to_numpy(X)
        self.predicts = []
        for tree in self.trees:
            self.predicts.append(tree.predict(X))
        self.pred_by_row = np.array(self.predicts).T
        
        ensemble_predict = []
        for row in self.pred_by_row:
            ensemble_predict.append(np.mean(row))
        return ensemble_predict
    
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