import numpy as np

def get_error(true,pred):
    return pred-true

def get_squared_error(true,pred):
    return np.square(get_error(true,pred))   

def mean_squared_error(true, pred):
    return np.mean(get_squared_error(true,pred))

def root_mean_squared_error(true,pred):
    np.sqrt(mean_squared_error(true,pred))

def mean_absolute_error(true,pred):
    return np.mean(get_error(true,pred))

def r2_score(true,pred):
    SSE = np.sum(get_squared_error(true,pred))
    SST = np.sum(get_squared_error(true,np.mean(true)*np.array(pred).shape[0]))
    return 1.-(SSE/SST)

def adj_r2(true, pred, num_data, num_features):
    rsquare = r2_score(true,pred)
    temp = (1-rsquare)*(num_data-1)
    temp = temp/(num_data-num_features-1)
    temp = 1 - temp
    return temp

def test_model_results(X, true, pred):
    print("Mean Squared Error: ", mean_squared_error(true,pred))
    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(true,pred)))
    print("Mean Absolute Error: ",mean_absolute_error(true,pred))
    r2 = r2_score(true,pred)
    print("R2: ", r2)
    try: 
        print("Adj R2: ", adj_r2(true,pred,X.shape[0],X.shape[1]))
    except: 
        print("Adj R2: ", adj_r2(true, pred, X.shape[0],1))
