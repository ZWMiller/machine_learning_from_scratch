import pandas as pd
import numpy as np
from collections import defaultdict

class bernoulli_naive_bayes:
    
    def __init__(self, smoothing = 1.):
        """
        Bernoulli Naive Bayes builds it's understanding of the data by
        applying Bayes rule and calculating the conditional probability of
        being a class based on a probabilistic understanding of how the 
        class has behaved before. We only care if a feature is zero or non-zero
        in this style of naive bayes and will calculate our conditional probabilities
        accordingly. 
        ---
        Inputs:
        smoothing: the Laplace smoothing factor overcome the problem of multiplying
        a 0 probability, that causes the total probability to be 0.
        """
        self._prob_by_class = defaultdict(float)
        self._cond_probs = defaultdict(lambda: defaultdict(float))
        self._log_prob_by_class = defaultdict(float)
        self._log_cond_probs = defaultdict(lambda: defaultdict(float))
        self._data_cols = None
        self._smoothing = smoothing
    
    def fit(self, X, y):
        """
        For each class, we find out what percentage of the data is that class.
        We then filter the data so only the rows that are that class remain,
        and then go column by column - calculating what percentage of rows are
        non-zero, given the class. We store all of these values to be used later 
        for predictions. We also store the log of these values for later prediction.
        ---
        Input: X, data (array/DataFrame)
        y, targets (array/Series)
        """
        X = self.convert_to_array(X)
        y = self.pandas_to_numpy(y) # keep as 1D
        self._data_cols = X.shape[1]
       
        self._classes = np.unique(y)
        
        for cl in self._classes:
            filtered_targets = y[y == cl]
            filtered_data = X[y == cl]
            self._prob_by_class[cl] = len(filtered_targets)/len(y)
            self._log_prob_by_class[cl] = np.log(self._prob_by_class[cl])
            denom = len(filtered_targets)
            for col in range(self._data_cols):
                binarized_column = filtered_data.T[col] > 0 
                num_ones = np.sum(binarized_column)
                #smoothing applied here so we never get a zero probability
                self._cond_probs[cl][col] = (num_ones+self._smoothing)/(denom+self._smoothing) 
                self._log_cond_probs[cl][col] = np.log(self._cond_probs[cl][col])
                
    def predict(self, X):
        """
        Wrapper to return only the class of the prediction
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict")
    
    def predict_proba(self, X):
        """
        Wrapper to return probability of each class of the prediction
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict_proba")
    
    def predict_log_proba(self, X):
        """
        Wrapper to return log of the probability of each class of 
        the prediction.
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict_log_proba")
    
    def _predict(self, X, mode="predict"):
        """
        For each data point, we go through and calculate the probability
        of it being each class. We do so by using the probability of
        seeing each value per feature, then combining them together with 
        the class probability. We work in the log space to fight against
        combining too many really small or large values and under/over 
        flowing Python's memory capabilities for a float. Depending on the mode
        we return either the prediction, the probabilities for each class,
        or the log of the probabilities for each class.
        ---
        Inputs: X, data (array/DataFrame)
        mode: type of prediction to return, defaults to single prediction mode
        """
        X = self.convert_to_array(X)
        X = (X > 0).astype(int) # convert to 1 or 0
        results = []
        for row in X:
            beliefs = []
            for cl in self._classes:
                prob_for_class = self._log_prob_by_class[cl]
                for col in range(self._data_cols):
                    p = self._log_cond_probs[cl][col]
                    # The row or (1-row) chooses either the 0 or 1 probability
                    # based on whether our row is a 0 or 1.
                    prob_for_class += p*row[col] + (1-p)*(1-row[col])
                beliefs.append([cl, prob_for_class])
            
            if mode == "predict_log_proba":
                _, log_probs = zip(*beliefs)
                results.append(log_probs)
            
            elif mode == "predict_proba":
                _, probs = zip(*beliefs)
                unlog_probs = np.exp(probs)
                normed_probs = unlog_probs/np.sum(unlog_probs)
                results.append(normed_probs)
            
            else:
                sort_beliefs = sorted(beliefs, key=lambda x: x[1], reverse=True)
                results.append(sort_beliefs[0][0])
        
        return np.array(results).reshape(-1,1)
    
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