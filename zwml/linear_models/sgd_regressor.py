import numpy as np
import pandas as pd

class sgd_regressor:
    
    def __init__(self, n_iter=100, alpha=0.01, verbose=False, return_steps=False, fit_intercept=True, 
                 dynamic=False, loss='ols', epsilon=0.1, regularize='L2', lamb=1e-6, l1_perc = 0.5):
        """
        Stochastic Gradient Descent Algorithm, with OLS cost function.
        ---
        KWargs:
        
        n_iter: number of epochs to run in while fitting to the data. Total number of steps
        will be n_iter*X.shape[0]. 
        alpha: The learning rate. Moderates the step size during the gradient descent algorithm.
        verbose: Whether to print out coefficient information during the epochs  
        return_steps: If True, fit returns a list of the coefficients at each update step for diagnostics   
        fit_intercept: If True, an extra coefficient is added with no associated feature to act as the
                       base prediction if all X are 0.                   
        dynamic: If true, an annealing scedule is used to scale the learning rate.    
        regularize: Choose what type, if any, of regularization to apply. Options are "L2" (Ridge),
                    "L1" (Lasso), and "EN" (Elastic Net: L1 + L2). All other inputs will not apply
                    regularization    
        lamb: Stands for lambda. Sets the strength of the regularization. Large lambda causes large
              regression. If regularization is off, this does not apply to anything.        
        l1_perc: If using elastic net, this variable sets what portion of the penalty is L1 vs L2. 
                 If regularize='EN' and l1_perc = 1, equivalent to regularize='L1'. If 
                 regularize='EN' and l1_perc = 0, equivalent to regulzarize='L2'.
        """
        self.coef_ = None
        self.trained = False
        self.n_iter = n_iter
        self.alpha_ = alpha
        self.verbosity = verbose
        self._return_steps = return_steps
        self._fit_intercept = fit_intercept
        self._next_alpha_shift = 0.1 # Only used if dynamic=True
        self._dynamic = dynamic
        self._regularize = regularize
        self._lamb = lamb
        self._l1_perc = l1_perc
        if self._l1_perc > 1. or self._l1_perc < 0.:
            raise ValueError("l1_perc must be between 0 and 1")
        
    def update(self, x, error):
        """
        Calculating the change of the coeficients for SGD. This is the derivative of the cost 
        function. B_i = B_i - alpha * dJ/dB_i. If fit_intercept=True, a slightly different 
        value is used to update the intercept coefficient, since the associated feature is "1."
        ---
        Inputs:
        
        data_point: A single row of the feature matrix. Since this is Stochastic, batches are not allowed.
        
        error: The residual for the current data point, given the current coefficients. Prediction - True
        for the current datapoint and coefficients.
        """
        if self._regularize == 'L2':
            step = self.alpha_*error*x + 2*self._lamb*self.coef_[1:]
        elif self._regularize == 'L1':
            step = self.alpha_*error*x + self._lamb*np.sign(self.coef_[1:])
        elif self._regularize == "EN":
            step = self.alpha_*error*x + 2*(1 - self._l1_perc)*self._lamb*self.coef_[1:] + self._l1_perc*self._lamb*np.sign(self.coef_[1:])
        else:
            step = self.alpha_*error*x
            
        # We don't regularize the intercept term. This term is adjusting for the "shift" in our
        # target data - and we don't want to shrink it, or we'll introduce bias.
        if self._fit_intercept:  
            self.coef_[1:] -= step
            self.coef_[0] -= self.alpha_ * error
        else:
            self.coef_ -= step
        
    def shuffle_data(self, X, y):
        """
        Given X and y, shuffle them together to get a new_X and new_y that maintain feature-target
        correlations. 
        ---
        Inputs:
        
        X: A numpy array of any shape
        y: A numpy array of any shape
        
        Both X and y must have the same first dimension length.
        
        Returns:
        X,y: two numpy arrays
        """
        assert len(X) == len(y)
        permute = np.random.permutation(len(y))
        return X[permute], y[permute]
        
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
    
    def dynamic_learning_rate_check(self, epoch):
        """
        If dynamic=True, shrink the learning rate by a factor of 2 after every 10% of
        the total number of epochs. This should cause a more direct path to the global 
        minimum after the initial large steps.
        ---
        Inputs: epoch (int,float), the current iteration number. 
        """
        percent_of_epochs = float(epoch)/float(self.n_iter)
        if percent_of_epochs > self._next_alpha_shift:
            self._next_alpha_shift += 0.1
            self.alpha_ = self.alpha_/2.
            
    def fit(self, X, y):
        """
        Actually trains the model. Given feature-target combinations, gradient descent is performed
        using the optimization stepping given in the 'update' function. At present, all epochs are 
        completed, as no tolerance is set. The learning rate is currently fixed.
        ---
        Inputs: 
            X (array, dataframe, series), The features to regress on using SGD
            y (array, series), Must be a 1D set of targets.
        Outputs:
            steps (optional): If return_steps=True, a list of the evolution of the coefficients is returned
        """
        X = self.convert_to_array(X)
        y = self.convert_to_array(y)
        self._stdy = np.std(y)
        self.coef_ = self.init_coef(X)
        if self._return_steps:
            steps = []
            steps.append(np.copy(self.coef_))
        for epoch in range(self.n_iter):
            shuf_X, shuf_y = self.shuffle_data(X,y)
            if self.verbosity:
                print("Epoch ", epoch, ", Coeff: ", self.coef_)
            for data, true in zip(shuf_X,shuf_y):
                pred = self.predict(data, is_array=True)
                error = pred - true
                self.update(data, error)
                if self._return_steps:
                    steps.append(np.copy(self.coef_))
            if self._dynamic:
                self.dynamic_learning_rate_check(epoch)
        if self._return_steps:
            return steps
            
    def init_coef(self, X):
        """
        Returns the initial starting values for the coefficients. At present, these are randomly
        set. If fit_intercept = True, an extra coefficient is generated. 
        ---
        Input: X, Feature matrix. Needed to decide how many coefficients to generate.
        """
        if self._fit_intercept:
            return np.random.rand(X.shape[1]+1)
        return np.random.rand(X.shape[1])

    def predict(self, X, is_array=False):  
        """
        Returns a prediction for a new data set, using the model coefficients.
        ---
        Input: 
            X (dataframe, array): The new feature set. Must be the same number of columns
            as the initial training features. 
        Output:
            prediction (array): The dot product of the input data and the coeficients.
        """
        if not is_array:
            X = self.convert_to_array(X)
        if not self.coef_.all():
            raise ValueError("Coefficients not defined, must fit() before predict().")
        if self._fit_intercept:
            return np.dot(X,self.coef_[1:]) + self.coef_[0]
        return np.dot(X,self.coef_)
    
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