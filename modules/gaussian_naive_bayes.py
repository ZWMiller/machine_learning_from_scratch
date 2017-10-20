import pandas as pd
import numpy as np
from collections import defaultdict

class gaussian_naive_bayes:
    
    def __init__(self):
        """
        Gaussian Naive Bayes builds it's understanding of the data by
        applying Bayes rule and calculating the conditional probability of
        being a class based on a probabilistic understanding of how the 
        class has behaved before. We will assume each feature is normally
        distributed in its own space, then use a gaussian PDF to calculate
        the probability of a class based on behavior. 
        """
        self._prob_by_class = defaultdict(float)
        self._cond_means = defaultdict(lambda: defaultdict(float))
        self._cond_std = defaultdict(lambda: defaultdict(float))
        self._log_prob_by_class = defaultdict(float)
        self._data_cols = None
        
    def gaus(self, x, mu=0, sig=1):
        """
        Returns the probability of x given the mean and standard
        deviation provided - assuming a Gaussian probability.
        ---
        Inputs: x (the value to find the probability for, float),
        mu (the mean value of the feature in the training data, float),
        sig (the standard deviation of the feature in the training data, float)
        Outputs: probability (float)
        """
        norm = 1/(np.sqrt(2*np.pi*sig**2))
        return norm*np.exp(-(x-mu)**2/(2*sig**2))
    
    def fit(self, X, y):
        """
        For each class, we find out what percentage of the data is that class.
        We then filter the data so only the rows that are that class remain,
        and then go column by column - calculating the mean and standard dev
        for the values of that column, given the class. We store all of these
        values to be used later for predictions.
        ---
        Input: X, data (array/DataFrame)
        y, targets (array/Series)
        """
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        if not self._data_cols:
            try: 
                self._data_cols = X.shape[1]
            except IndexError:
                self._data_cols = 1
        X = self.check_feature_shape(X)
        
        self._classes = np.unique(y)
        
        for cl in self._classes:
            self._prob_by_class[cl] = len(y[y == cl])/len(y)
            self._log_prob_by_class[cl] = np.log(self._prob_by_class[cl])
            filt = (y == cl)
            filtered_data = X[filt]
            for col in range(self._data_cols):
                self._cond_means[cl][col] = np.mean(filtered_data.T[col])
                self._cond_std[cl][col] = np.std(filtered_data.T[col])
                
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
        of it being each class. We do so by sampling the probability of
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
        X = self.pandas_to_numpy(X)
        X = self.check_feature_shape(X)
        results = []
        for row in X:
            beliefs = []
            for cl in self._classes:
                prob_for_class = self._log_prob_by_class[cl]
                for col in range(self._data_cols):
                    if self._cond_std[cl][col]:
                        p = self.gaus(row[col],mu=self._cond_means[cl][col],sig=self._cond_std[cl][col])
                        logp = np.log(p)
                        prob_for_class += logp
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
        
        return results
    
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
      
    def check_feature_shape(self, X):
        """
        Helper function to make sure any new data conforms to the fit data shape
        ---
        In: numpy array, (unknown shape)
        Out: numpy array, shape: (rows, self.data_cols)"""
        return X.reshape(-1,self._data_cols)
            
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return np.array(x)
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)