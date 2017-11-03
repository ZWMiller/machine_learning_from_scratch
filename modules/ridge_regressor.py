import numpy as np
import pandas as pd
import sys
from sgd_regressor import sgd_regressor

class ridge_regressor(sgd_regressor):
    
    def __init__(self, n_iter=100, alpha=0.01, verbose=False, return_steps=False, fit_intercept=True, 
                 dynamic=True, loss='ols', epsilon=0.1, lamb=1e-6, l1_perc = 0.5):
        """
        Ridge Regressor - This is a wrapper on the SGD class where the regularization is set
        to the L2 Norm. All other functionality is the same as the SGD class.
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
        self._regularize = 'L2'
        self._lamb = lamb
        self._l1_perc = l1_perc