import numpy as np

class OLS:
    
    def __init__(self, w_intercept=True):
        self.coef_ = None
        self.intercept = w_intercept
        self.is_fit = False
        
    def add_intercept(self,X):
        """
        Adds an 'all 1's' bias term to function as the y-intercept
        """
        if type(X) == type(np.array([5])):
            rows = X.shape[0]
        else:
            X = np.array([[X]])
            rows = 1
        inter = np.ones(rows).reshape(-1,1)
        return np.hstack((X,inter))
        
    def fit(self, X, y):
        """
        Read in X (all features) and y (target) and use the Linear Algebra solution
        to extract the coefficients for Linear Regression.
        """
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        if self.intercept:
            X = self.add_intercept(X)
        temp_xtx = np.linalg.inv(np.dot(X.T,X))
        temp_xty = np.dot(X.T,y)
        self.coef_ = np.dot(temp_xtx,temp_xty)
        self.is_fit = True
    
    def predict(self,X):
        """
        Takes in a new X value (that must be the same shape as the original X for fitting)
        and returns the predicted y value, using the coefficients from fitting.
        """
        if not self.is_fit:
            raise ValueError("You have to run the 'fit' method before using predict!")
        if type(X) == type([5]):
            X = np.array(X)
        if type(X) == type(5) or type(X) == type(5.):
            X = np.array([X])
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.intercept:
            X = self.add_intercept(X)
        return np.dot(X,self.coef_)[0][0]
    
    def score(self, X, true):
        """
        Takes in X, y pairs and measures the performance of the model.
        ---
        Inputs: X, y (features, labels; np.arrays)
        Outputs: Mean Squared Error (float)
        """
        pred = self.predict(X)
        mse = np.mean(np.square(true-pred))
        return mse
