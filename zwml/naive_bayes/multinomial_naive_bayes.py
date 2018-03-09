import pandas as pd
import numpy as np
from collections import defaultdict

class multinomial_naive_bayes:
    
    def __init__(self, smoothing = 1.):
        """
        Multinomial Naive Bayes builds it's understanding of the data by
        applying Bayes rule and calculating the conditional probability of
        being a class based on a probabilistic understanding of how the 
        class has behaved before. We calculate conditional probabilities
        . 
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
        and then go column by column - calculating what of total counts in the
        class come from that feature. We store all of these values to be used later 
        for predictions. We also store the log of these values for later prediction.
        ---
        Input: X, data (array/DataFrame)
        y, targets (array/Series)
        """
        X = self.convert_to_array(X)
        y = self.pandas_to_numpy(y)
        self._data_cols = X.shape[1]
       
        self._classes = np.unique(y)
        
        for cl in self._classes:
            filtered_targets = y[y == cl]
            filtered_data = X[y == cl]
            self._prob_by_class[cl] = len(filtered_targets)/len(y)
            self._log_prob_by_class[cl] = np.log(self._prob_by_class[cl])
            denom = np.sum(filtered_data)
            for col in range(self._data_cols):
                sum_of_column = np.sum(filtered_data.T[col])
                #smoothing applied here so we never get a zero probability
                self._cond_probs[cl][col] = (sum_of_column+self._smoothing)/(denom+self._smoothing) 
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
        seeing each feature/class and multiplying that by the number
        of times we see that feature, then combining them together with 
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
        results = []
        for row in X:
            beliefs = []
            for cl in self._classes:
                prob_for_class = self._log_prob_by_class[cl]
                for col in range(self._data_cols):
                    val = row[col]
                    p = self._log_cond_probs[cl][col]
                    prob_for_class += val*p
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