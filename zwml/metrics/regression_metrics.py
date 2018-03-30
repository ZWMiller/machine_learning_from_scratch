import numpy as np
import pandas as pd

def get_error(true,pred):
    """
    Returns predicted - true for each entry
    """
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return pred-true

def get_square_error(true,pred):
    """
    Returns the square of predicted - true for each entry
    """
    return np.square(get_error(true,pred))   

def mean_square_error(true, pred):
    """
    Returns the average predicted - true
    """
    return np.mean(get_square_error(true,pred))

def root_mean_square_error(true,pred):
    """
    Returns the sqrt of mean square error
    """
    return np.sqrt(mean_square_error(true,pred))

def mean_absolute_error(true,pred):
    """
    Returns the mean absolute value of error
    """
    return np.mean(np.abs(get_error(true,pred)))

def sum_square_error(true,pred):
    """
    Returns the sum of squared errors
    """
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return np.sum(get_square_error(true,pred))

def r2_score(true,pred):
    """
    Returns R2 which is computed by
    SSE = sum of squared errors from the model
    SST = sume of squared errors to the mean of the data (y)
    R2 = 1 - SSE/SST
    """
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    SSE = np.sum(get_square_error(true,pred))
    shpe = len(np.array(true))
    SST = np.sum(get_square_error(true,np.mean(true)*shpe))
    return 1.-(SSE/SST)

def adj_r2(true, pred, X):
    """
    Returns a version of R2 that penalizes for having many
    features. Fights against false correlations in data
    and is generally better than R2.
    """
    X = pandas_to_numpy(X)
    rsquare = r2_score(true,pred)
    num_data = X.shape[0]
    num_features = X.shape[1]
    temp = (1-rsquare)*(num_data-1)
    temp = temp/(num_data-num_features-1)
    temp = 1 - temp
    return temp

def assess_model(true, pred):
    """
    Computes a suite of metrics all at once
    """
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return sum_square_error(true,pred), mean_square_error(true,pred), root_mean_square_error(true,pred)

def test_regression_results(X, true, pred):
    """
    A print out of many of the metrics that show model performance
    """
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    print("Mean Square Error: ", mean_square_error(true,pred))
    print("Root Mean Square Error: ", np.sqrt(mean_square_error(true,pred)))
    print("Mean Absolute Error: ",mean_absolute_error(true,pred))
    r2 = r2_score(true,pred)
    print("R2: ", r2)
    print("Adj R2: ", adj_r2(true,pred,X))

def pandas_to_numpy(x):
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