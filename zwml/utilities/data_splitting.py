import numpy as np
import pandas as pd

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

def train_test_split_verbose(X, y, test_size=0.3, seed=None):
    """
    Takes in features and labels and returns X_train, X_test, y_train, and y_test
    ----
    In: X (features), y (labels), test_size (percentage of data to go into test), seed
    Out: X_train, X_test, y_train, and y_test
    """
    if seed:
        np.random.seed(seed)

    X = pandas_to_numpy(X)
    y = pandas_to_numpy(y)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    splitter = np.random.choice([0,1],size=y.shape,p=[1-test_size,test_size])
    for x,y,z in zip(X,y,splitter):
        if z == 0:
            X_train.append(x)
            y_train.append(y)
        else:
            X_test.append(x)
            y_test.append(y)
    return X_train, X_test, y_train, y_test

def train_test_split(X, y, test_size=0.3, seed=None):
    """
    Takes in features and labels and returns X_train, X_test, y_train, and y_test.
    If test_size is a float, it acts as a percentage of data to become test_size.
    If test_size is an int, that many records are returned as a test.
    ----
    In: X (features), y (labels), test_size (percentage of data to go into test), seed
    Out: X_train, X_test, y_train, and y_test
    """
    assert len(X) == len(y), "Length of records and labels must be equal!"
    X = pandas_to_numpy(X)
    y = pandas_to_numpy(y)

    if isinstance(test_size, float):
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be an int, or between 0 and 1")
        test_size = int(len(y)*test_size)
    elif not isinstance(test_size, int):
        raise TypeError("test_size must be an int or a float")

    if seed:
        np.random.seed(seed)

    permute = np.random.permutation(len(y))
    X = X[permute]
    y = y[permute]
    return X[test_size:], X[:test_size], y[test_size:], y[:test_size]

import sys 
sys.path.append('../.')
from zwml.metrics.regression_metrics import * 

class cross_val:
    def __init__(self, show_plot=False, feat_num=0):
        """
        The Cross-Val object contains several objects that the user may want to 
        use later, including a final copy of the best model.
        ---
        Params:
        show_plot: should it plot the data showing the splits
        feat_num: if show_plot, which feature should be used (by column num)
        best_model: the model with the lowest MSE is kept for later usage
        """
        self.show_plot = show_plot
        self.feat_num = feat_num
        self.best_model = None
        self.best_model_score = None

    def plot_single_feature_vs_label(self, X_train, X_test, y_train, y_test, feature_num=0, 
            title="Checking Train-Test Split"):
        """
        This helper method is to make plots of the data being split 
        with one feature vs the target label, showing each fold for 
        visual inspection of the splits. 
        """
        x_plot = []
        x_plot_test = []
        for j in X_train:
            x_plot.append(j[feature_num])
        for j in X_test:
            x_plot_test.append(j[feature_num])

        plt.figure(figsize=(8,6))
        plt.scatter(x_plot, y_train, c='b')
        plt.scatter(x_plot_test, y_test, c='r')
        plt.xlabel("Feature " + str(feature_num))
        plt.ylabel("Y");
        plt.title(title);

    def plot_coefs(self):
        """
        This method shows the coefficient values for each fold in a plot.
        If there are 10 coefficient, there will be 10 plots. If there were 3
        folds, each plot will contain 3 points.
        """
        if not self.coefs:
            print("Either your model doesn't have coefficients, or you")
            print("must run cross_validation_scores first!")
            return            
        for coef in range(len(self.coefs[0])):
            plot_x = []
            plot_y = []
            i=1
            for fold in self.coefs:
                plot_x.append(i)
                plot_y.append(fold[coef])
                i+=1
            plt.figure(figsize=(10,8))
            plt.plot(plot_x,plot_y)
            plt.plot(plot_x,[np.mean(plot_y)]*len(plot_x),'r--')
            plt.ylabel("coef "+str(coef))
            plt.xlabel("Fold ID")
            plt.xticks([x for x in range(1,FOLDS+1)])
            plt.title("Variation of Coefficient Across Folds")

    def cross_validation_scores_verbose(self, model, X, y, k=5, random_seed=42):
        """
        Splits the dataset into k folds by randomly assigning each row a
        fold ID. Afterwards, k different models are built with each fold being
        left out once and used for testing the model performance.
        ---
        Inputs:
        model: must be a class object with fit/predict methods. 
        X: feature matrix (array)
        y: labels (array)
        k: number of folds to create and use
        random_seed: sets the random number generator seed for reproducibility
        """
        X = pandas_to_numpy(X)
        y = pandas_to_numpy(y)
        self.score_folds = []
        coefs = []
        fold_nums = [x for x in range(k)]
        np.random.seed(random_seed)
        splitter = np.random.choice(fold_nums,size=y.shape)
        best_score = None
        for fold in fold_nums:
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for x2,y2,z2 in zip(X,y,splitter):
                if z2 == fold:
                    X_test.append(x2)
                    y_test.append(y2)
                else:
                    X_train.append(x2)
                    y_train.append(y2)
            model.fit(X_train,y_train)
            current_score = model.score(X_test, y_test)
            self.score_folds.append(current_score)
            if not best_score or current_score > best_score:
                best_score = current_score
                self.best_model = model
                self.best_model_score = current_score
            if model.coef_.any():
                coefs.append(model.coef_)
            if self.show_plot:
                plot_title = "CV Fold " + str(fold)
                plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=self.feat_num, 
                        title=plot_title)
                if coefs:
                    self.coefs = coefs     

    def cross_validation_scores(self, model, X, y, k=5, random_seed=42):
        """
        Splits the dataset into k folds by randomly assigning each row a
        fold ID. Afterwards, k different models are built with each fold being
        left out once and used for testing the model performance.
        ---
        Inputs:
        model: must be a class object with fit/predict methods. 
        X: feature matrix (array)
        y: labels (array)
        k: number of folds to create and use
        random_seed: sets the random number generator seed for reproducibility
        """
        X = pandas_to_numpy(X)
        y = pandas_to_numpy(y)
        self.score_folds = []
        coefs = []
        fold_nums = [x for x in range(k)]
        np.random.seed(random_seed)
        splitter = np.random.choice(fold_nums,size=y.shape)
        best_score = None
        permute = np.random.permutation(len(y))
        X = X[permute]
        y = y[permute]
        test_size = len(y)//k
        permute = np.random.permutation(len(y))
        for fold in range(k):
            start = fold*test_size
            end = fold*test_size+test_size
            X_test = X[permute[start:end]]
            y_test = y[permute[start:end]]
            X_train = X[~permute[start:end]]
            y_train = y[~permute[start:end]]
            
            model.fit(X_train,y_train)
            current_score = model.score(X_test, y_test)
            self.score_folds.append(current_score)
            if not best_score or current_score > best_score:
                best_score = current_score
                self.best_model = model
                self.best_model_score = current_score
            if hasattr(model, 'coef_'):
                coefs.append(model.coef_)
            if self.show_plot:
                plot_title = "CV Fold " + str(fold)
                plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=self.feat_num, 
                        title=plot_title)
                if coefs:
                    self.coefs = coefs     

    def print_report(self):
        """
        After the CV has been run, this method will print some summary statistics
        as well as the coefficients from the model.
        """
        print("Mean Score: ", np.mean(self.score_folds))
        print("Score by fold: ", self.score_folds)
        if self.coefs:
            print("Coefs (by fold): ")
            for i,c in enumerate(self.coefs):
                print("Fold ",i,": ",c)
