import numpy as np
import pandas as pd

def get_error(true,pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return pred-true

def get_square_error(true,pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return np.square(get_error(true,pred))   

def mean_square_error(true, pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return np.mean(get_square_error(true,pred))

def root_mean_square_error(true,pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return np.sqrt(mean_square_error(true,pred))

def mean_absolute_error(true,pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return np.mean(get_error(true,pred))

def sum_square_error(true,pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return np.sum(get_square_error(true,pred))

def r2_score(true,pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    SSE = np.sum(get_square_error(true,pred))
    shpe = len(np.array(true))
    SST = np.sum(get_square_error(true,np.mean(true)*shpe))
    return 1.-(SSE/SST)

def adj_r2(true, pred, num_data, num_features):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    rsquare = r2_score(true,pred)
    temp = (1-rsquare)*(num_data-1)
    temp = temp/(num_data-num_features-1)
    temp = 1 - temp
    return temp

def assess_model(true, pred):
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    return sum_square_error(true,pred), mean_square_error(true,pred), root_mean_square_error(true,pred)

def test_regression_results(X, true, pred):
    X = pandas_to_numpy(X)
    true = pandas_to_numpy(true)
    pred = pandas_to_numpy(pred)
    print("Mean Square Error: ", mean_square_error(true,pred))
    print("Root Mean Square Error: ", np.sqrt(mean_square_error(true,pred)))
    print("Mean Absolute Error: ",mean_absolute_error(true,pred))
    r2 = r2_score(true,pred)
    print("R2: ", r2)
    try: 
        print("Adj R2: ", adj_r2(true,pred,X.shape[0],X.shape[1]))
    except: 
        print("Adj R2: ", adj_r2(true, pred, X.shape[0],1))

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
