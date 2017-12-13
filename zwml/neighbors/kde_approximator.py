import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy

class kde_approximator:
    
    def __init__(self,kernel='gaus', bandwidth=1., grid_fineness=10):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.estim = self.gaus
        self.data_cols = None
        self.grid_fineness = grid_fineness
        
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
        diff = np.sqrt(np.sum((x-mu)**2))
        norm = 1/(np.sqrt(2*np.pi*sig**2))
        return norm*np.exp(-diff**2/(2*sig**2))
    
    def get_grid(self, X):
        if not self.data_cols:
            try: 
                self.data_cols = X.shape[1]
            except IndexError:
                self.data_cols = 1
        mins, maxes = [], []
        
        for col in range(self.data_cols): 
            data = X.T[col]
            mins.append(np.min(data)-abs(np.min(data)*0.10))
            maxes.append(np.max(data)+abs(np.max(data)*0.10))
        grid = np.stack(np.meshgrid(*[np.linspace(i,j,self.grid_fineness) for i,j in zip(mins, maxes)], indexing='ij'),self.data_cols)
        return grid
    
    def fit(self, X):
        """
        ---
        In: X (features), np.array or pandas dataframe/series
        """
        X = self.pandas_to_numpy(X)
        if not self.data_cols:
            try: 
                self.data_cols = X.shape[1]
            except IndexError:
                self.data_cols = 1
                
        X = self.check_feature_shape(X)
        self.X = copy(X)
    
    def make_surface(self):
        """
        ---
        In: X (features), np.array or pandas dataframe/series
        """
        X = self.X
        
        if not self.data_cols:
            try: 
                self.data_cols = X.shape[1]
            except IndexError:
                self.data_cols = 1
                
        X = self.check_feature_shape(X) 
        span = self.get_grid(X)
        
        probs = []
        points = []
        for dim in span:
            for p in dim:
                prob = 0.
                for d in X: 
                    prob += self.estim(p,mu=d,sig=self.bandwidth)
                if np.isnan(prob):
                    prob = 0.
                points.append(p)
                probs.append(prob)
        self.region = points 
        self.probs = probs
        
    def sample(self, num_samples=1, random_state=None):
        if random_state:
            np.random.seed(random_state)
        
        samples = []
        for i in range(num_samples):
            pt = self.X[np.random.randint(self.X.shape[0])]
            sample_pt = []
            for dim in pt:
                sample_pt.append(np.random.normal(dim, self.bandwidth))
            samples.append(sample_pt)
        return np.array(samples)
    
    def check_feature_shape(self, x):
        """
        Helper function to make sure any new data conforms to the fit data shape
        ---
        In: numpy array, (unknown shape)
        Out: numpy array, shape: (rows, self.data_cols)"""
        return x.reshape(-1,self.data_cols)
    
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
    
    def make_plot(self):
        if self.data_cols == 2:
            Xpl, Ypl  = zip(*self.region)
            Zpl = kde2.probs/max(self.probs)
            fig = plt.figure(dpi=200, figsize=(18,14))
            ax = fig.gca(projection='3d')
            ax.plot_trisurf(Xpl,Ypl,Zpl, cmap=plt.cm.rainbow, linewidth=1)

            Xsc, Ysc = zip(*X)
            ax.scatter(Xsc,Ysc,[max(Zpl)]*len(Xsc),c='k',s=20, label="Data", alpha=0.5);
            proxy = plt.Circle((0,0), fc="k")
            ax.legend([proxy],['Data (z = 1)'], fontsize=18, loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#333333');
            ax.set_zlabel("Norm. Prob.",fontsize=16, labelpad=10)
            ax.set_xlabel("X",fontsize=16, labelpad=10)
            ax.set_ylabel("Y",fontsize=16, labelpad=10);
        
        elif self.data_cols == 1:
            plt.figure(figsize=(10,6))
            plt.hist(X, label="Binned data", bins=18, alpha=0.8, zorder=1)
            plt.plot(self.region, self.probs, c='k', lw=3, label="KDE", zorder=2);
            plt.scatter(X, [5]*len(X), marker='o', c='r', s=30, alpha=0.3,label='Actual Data', zorder=3)
            plt.legend(fontsize=20, loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#333333');
            ax = plt.gca()
        else:
            print("Can only draw if KDE is done on 2 or fewer columns.")
            return None
        return ax
        
    