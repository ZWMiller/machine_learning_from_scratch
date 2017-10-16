import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.3):
    """
    Takes in features and labels and returns X_train, X_test, y_train, and y_test
    ----
    In: X (features), y (labels), test_size (percentage of data to go into test)
    Out: X_train, X_test, y_train, and y_test
    """
    X = np.array(X)
    y = np.array(y)
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


from stats_regress import *

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
        X = np.array(X)
        y = np.array(y)
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
