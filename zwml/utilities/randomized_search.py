from itertools import product

class randomized_search():
    
    def __init__(self, model_name, param_grid, n_iters=10):
        self._base_model = str(model_name).replace(')','')
        #param grid format (gaus, 0, 10)
        self._param_grid = param_grid
        self.n_iters = int(n_iters)
        self.models = self.get_models()  
        
    def get_models(self):
        command_list = []
        for _ in range(self.n_iters):
            cmd = self._base_model
            if cmd[-1] != '(':
                cmd+=', '
            for key, value in self._param_grid.items():
                cmd += str(key)+"="+str(self.sampling(value))+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
    def sampling(self, params):
        dist = params[0]
        if dist == 'gaus':
            try:
                return np.random.normal(params[1],params[2])
            except:
                raise ValueError("'gaus' must have 2 parameters, a mean and sigma.")
        elif dist == 'uni':
            try:
                return np.random.uniform(params[1],params[2])
            except:
                raise ValueError("'uni' must have 2 parameters, a low and a high.")
        elif dist == 'randint':
            try:
                return np.random.randint(params[1],params[2])
            except:
                raise ValueError("'randint' must have 2 parameters, a low and a high.")
        elif dist == 'binom':
            try:
                return np.random.binomial(1,params[1])
            except:
                raise ValueError("'binom' must have 1 parameter, the probability")
        else:
            raise ValueError("Distribution must be in ['gaus', 'uni','randint','binom']")
            
    def fit(self, X, y):
        results = []
        for model_name in self.models:
            model = eval(model_name)
            model.fit(X,y)
            s = model.score(X,y)
            results.append([model, s, model_name])
        self.all_results = sorted(results, key=lambda x: x[1], reverse=True)
        self.best_model = self.all_results[0][0]
        self.best_score = self.all_results[0][1]
        
    def print_results(self):
        if self.all_results:
            print("Model    |    Score\n--------------------\n")
            for result in self.all_results:
                print(result[2], "   |   ", result[1],"\n")

                
from zwml.utilities import cross_val

class randomized_search_cv():
    
    def __init__(self, model_name, param_grid, n_iters=10, k=5):
        self._base_model = str(model_name).replace(')','')
        #param grid format (gaus, 0, 10)
        self._param_grid = param_grid
        self.n_iters = int(n_iters)
        self.k = k
        self.models = self.get_models()  
        
    def get_models(self):
        command_list = []
        for _ in range(self.n_iters):
            cmd = self._base_model
            if cmd[-1] != '(':
                cmd+=', '
            for key, value in self._param_grid.items():
                cmd += str(key)+"="+str(self.sampling(value))+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
    def sampling(self, params):
        dist = params[0]
        if dist == 'gaus':
            try:
                return np.random.normal(params[1],params[2])
            except:
                raise ValueError("'gaus' must have 2 parameters, a mean and sigma.")
        elif dist == 'uni':
            try:
                return np.random.uniform(params[1],params[2])
            except:
                raise ValueError("'uni' must have 2 parameters, a low and a high.")
        elif dist == 'randint':
            try:
                return np.random.randint(params[1],params[2])
            except:
                raise ValueError("'randint' must have 2 parameters, a low and a high.")
        elif dist == 'binom':
            try:
                return np.random.binomial(1,params[1])
            except:
                raise ValueError("'binom' must have 1 parameter, the probability")
        else:
            raise ValueError("Distribution must be in ['gaus', 'uni','randint','binom']")
    
    def fit(self, X, y):
        results = []
        for model_name in self.models:
            model = eval(model_name)
            cv = cross_val()
            cv.cross_validation_scores(model, X, y, self.k)
            results.append([model, cv.score_folds, model_name])
        self.all_results = sorted(results, key=lambda x: np.mean(x[1]), reverse=True)
        self.best_model = self.all_results[0][0]
        self.best_score = self.all_results[0][1]
        
    def print_results(self, coefs=False, mean=False):
        if self.all_results:
            print("Model    |    Scores\n--------------------")
            for result in self.all_results:
                if mean:
                    print(result[2], "   |   ", np.mean(result[1]))
                else:
                    print(result[2], "   |   ", result[1])
                if coefs:
                    try:
                        print("Coefs: ", result[0].coefs_)
                    except AttributeError:
                        print("No Coefficients in model!")    
                print()