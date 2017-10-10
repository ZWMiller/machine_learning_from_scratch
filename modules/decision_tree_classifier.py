import math
import numpy as np

class decision_tree_classifier:
    
    def __init__(self, max_depth = None):
        self.tree = self.tree_split()
        self.data_cols = None
        self.max_depth = max_depth
        self.current_depth = 0
    
    # Sub class for handling recursive nodes (only makes sense in the scope of a tree)
    class tree_split:
        """
        A sub class for handling recursive nodes. Each node will contain the value and column
        for the current split, as well as links to the resulting nodes from the split. The 
        results attribute remains empty unless the current node is a leaf. 
        """
        def __init__(self,col=-1,value=None,results=None,label=None,tb=None,fb=None):
            self.col=col # column index of criteria being tested
            self.value=value # vlaue necessary to get a true result
            self.results=results # dict of results for a branch, None for everything except endpoints
            self.tb=tb # true decision nodes 
            self.fb=fb # false decision nodes
    
    def split_data(self, X, y, colnum, value):
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
        set1Y = y[split1]
        set2X = X[split2]
        set2Y = y[split2]
        return set1X, set1Y, set2X, set2Y

    def count_target_values(self, data):
        """
        Returns: A dictionary of target variable counts in the data
        """
        results = {}
        for row in data:
            if row not in results:
                results[row] = 0
            results[row] += 1
        return results

    def entropy(self, y):
        """
        Returns: Entropy of the data set, based on target values. 
        ent = Sum(-p_i Log(p_i), i in unique targets) where p is the percentage of the
        data with the ith label.
        Sidenote: We're using entropy as our measure of good splits. It corresponds to 
        information gained by making this split. If the split results in only one target type
        then the entropy new sets entropy is 0. If it results in a ton of different targets, the
        entropy will be high. 
        """
        results = self.count_target_values(y)
        log2=lambda x:math.log(x)/math.log(2)
        ent=0.
        for r in results.keys():
            p=float(results[r])/len(y) 
            ent-=p*log2(p)
        return ent  
    
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
    
    def check_feature_shape(self, x):
        """
        Helper function to make sure any new data conforms to the fit data shape
        ---
        In: numpy array, (unknown shape)
        Out: numpy array, shape: (rows, self.data_cols)"""
        return x.reshape(-1,self.data_cols)
    
    def fit(self, X, y):
        """
        Helper function to wrap the fit method. This makes sure the full nested 
        recursively built tree gets assigned to the correct variable name and 
        persists after training.
        """
        self.tree = self._fit(X,y)
    
    def _fit(self, X, y):
        """
        Builds the decision tree via a greedy approach, checking every possible
        branch for the best current decision. Decision strength is measured by
        information gain/entropy reduction. If no information gain is possible,
        sets a leaf node. Recursive calls to this method allow the nesting.
        ---
        Input: X (feature matrix), y (labels)
        Output: A nested tree built upon the node class."""
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        if not self.data_cols:
            self.data_cols = X.shape[1]
        self.check_feature_shape(X)
        if len(X) == 0: return tree_split()
        current_score = self.entropy(y)

        best_gain = 0.0
        best_criteria = None
        best_sets = None
        
        cols = X.shape[1]
        
        
        # Here we go through column by column and try every possible split, measuring the
        # information gain. We keep track of the best split then use that to send the split
        # data sets into the next phase of splitting.
        
        for col in range(cols):
            
            # find different values in this column
            column_values = set(X.T[col])
            # for each possible value, try to divide on that value
            for value in column_values:
                set1, set1_y, set2, set2_y = self.split_data(X, y, col, value)

                # Information gain
                p = float(len(set1)) / len(y)
                gain = current_score - p*self.entropy(set1_y) - (1-p)*self.entropy(set2_y)
                if gain > best_gain and len(set1_y) and len(set2_y):
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (np.array(set1), np.array(set1_y), np.array(set2), np.array(set2_y))
        
        
        # Now decide whether it's an endpoint or we need to split again.
        if (self.max_depth and self.current_depth < self.max_depth) or not self.max_depth:
            if best_gain > 0:
                self.current_depth += 1
                true_branch = self._fit(best_sets[0], best_sets[1])
                false_branch = self._fit(best_sets[2], best_sets[3])
                return self.tree_split(col=best_criteria[0], value=best_criteria[1],
                        tb=true_branch, fb=false_branch)
            else:
                return self.tree_split(results=self.count_target_values(y))
        else:
            return self.tree_split(results=self.count_target_values(y))

    def print_tree(self, indent="---"):
        """
        Helper function to make sure the correct tree gets printed.
        ---
        In: indent (how to show splits between nodes)
        """
        self._print_tree_(self.tree, indent)
    
    def _print_tree_(self, tree, indent):
        """
        Goes through node by node and reports the column and value used to split
        at that node. All sub-nodes are drawn in sequence below the node.
        """
        if tree.results: # if this is a end node
            print(str(tree.results))
        else:
            print('Column ' + str(tree.col)+' : '+str(tree.value)+'? ')
            # Print the branches
            print(indent+' True: ', end=' ')
            next_indent = indent+indent
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
        return np.array(results)
            
    def _predict(self, newdata, tree):
        """
        Uses the reusive structure of the tree to follow each split for
        a new data point. If the node is an endpoint, the available classes
        are sorted by "most common" and then the top choice is returned.
        """
        newdata = self.pandas_to_numpy(newdata)
        if tree.results: # if this is a end node
            return sorted(list(tree.results.items()), key=lambda x: x[1],reverse=True)[0][0]

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

    def score(self, X, y):
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
