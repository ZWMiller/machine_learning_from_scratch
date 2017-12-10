import numpy as np

class make_regression:
    
    def __init__(self):
        self.model_params = None
        self.num_feats = None
        self.random_state = None
        self.is_clean = None
        self.noise = None
        self.bias = None
        
    def build_dataset(self, num_feat=10, num_rows=100, random_state = None, num_important=10, 
                      noise=0.1, bias=None, dirty_data=False):
        assert num_feat > 0 and num_rows > 0, "Must have rows and features > 0."
        if random_state:
            np.random.seed(random_state)
            self.random_state = random_state

        means = np.random.uniform(-1,1, size=num_feat)
        sigmas = np.random.uniform(1e-6,1, size=num_feat)
        X = np.zeros((num_rows, num_feat))
        for i, mu in enumerate(means):
            X.T[i] = np.random.normal(mu, sigmas[i], num_rows)

        if bias == True:
            bias = np.random.uniform(-1,1)
        elif isinstance(bias, float):
            pass
        else:
            bias = 0.
        
        self.bias = bias

        if num_important > num_feat:
            num_important = num_feat
            
        self.num_important = num_important
        self.num_feats = num_feat

        target_builder = np.random.choice(np.arange(num_feat),num_important, replace=False)
        X_target = X.T[target_builder].T
        betas = np.random.uniform(-1,1,num_important)
        params = []
        for i,j in zip(betas, target_builder):
            params.append((j,i))
        self.model_params = params
        
        y = np.sum(X_target*betas, axis=1) + bias + np.random.normal(0, noise, num_rows)
        
        if dirty_data:
            X = self.muck_up_data(X)
            
        return X, y
    
    def muck_up_data(self, X, dup_cols=True, add_nan=True):
        if dup_cols:
            X = self.add_duplicate_columns(X)
        if add_nan:
            X = self.add_nans(X, add_nan)
        return X
    
    def add_duplicate_columns(self,X):
        cols_to_dup = np.random.choice(np.arange(self.num_feats), np.random.randint(1,self.num_feats), replace=False)
        new_X = np.hstack((X, X.T[cols_to_dup].T.reshape(-1,len(cols_to_dup))))
        return new_X
            
    def add_nans(self, X, add_nan_val):
        
        if isinstance(add_nan_val, float):
            num_of_nans = int(add_nan_val*X.size)   
        elif isinstance(add_nan_val, int):
            num_of_nans = add_nan_val
        else:
            max_nans = int(0.1*X.size)
            num_of_nans = np.random.randint(1,max_nans)
        print(num_of_nans)
        for _ in range(num_of_nans):
            i = np.random.randint(0,X.shape[0])
            j = np.random.randint(0,X.shape[1])
            X[i,j] = np.nan
        return X