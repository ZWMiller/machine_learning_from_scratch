from itertools import product

class randomized_search():
    
    def __init__(self, model_name, param_grid, n_iters=10):
        """
        Given a base model and a parameter grid of params
        for that model, goes through n_iters iterations and 
        builds a model with a sample from the provided params,
        and returns the score of the model.
        ---
        Inputs:
        model_name : the name of the model with parenthesis 
        and as a string. Any parameters you wish to set for all
        models can be set in the parameter name.
        param_grid: dictionary with parameter names as keys,
        and the appropriate values for parameter. 
            Param Options:
            ('gaus', mean, std dev) # normal with mean and std dev
            ('uni', low, high) # uniform between low and high
            ('randint', low, high) # int between low and high
            ('binom', p) # binomial sample with prob p
            ('choose', list) #picks one of the list
        n_iters: how many models to build with randomized params
        """
        self._base_model = str(model_name).replace(')','')
        self._param_grid = param_grid
        self.n_iters = int(n_iters)
        self.models = self.get_models()  
        
    def get_models(self):
        """
        For every iteration of the model, goes parameter
        by parameter and samples the distribution provided
        for the parameter. Creates a string that is the model
        name with the parameters as part of the string.
        """
        command_list = []
        for _ in range(self.n_iters):
            cmd = self._base_model
            if cmd[-1] != '(':
                cmd+=', '
            for key, value in self._param_grid.items():
                if value[0] == "choose":
                    cmd += str(key)+"='"+str(self.sampling(value))+"', "
                else:
                    cmd += str(key)+"="+str(self.sampling(value))+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
    def sampling(self, params):
        """
        Given a parameter set in the format (type, args...),
        choose a value using numpy's distributions.
        Param Options:
            ('gaus', mean, std dev) # normal with mean and std dev
            ('uni', low, high) # uniform between low and high
            ('randint', low, high) # int between low and high
            ('binom', p) # binomial sample with prob p
            ('choose', list) #picks one of the list
        """
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
        elif dist == 'choose':
            try:
                return np.random.choice(params[1])
            except:
                raise ValueError("'choose' must have 1 parameter, the list of possibilities")
        else:
            raise ValueError("Distribution must be in ['gaus', 'uni','randint','binom','choose']")
            
    def fit(self, X, y):
        """
        Uses the "eval" function in Python to convert the model
        name from string to an actual model. Fits each model
        and scores it. Creates a lists of models and scores.
        Sets the best possible model and score to be easily
        retrievable and usable.
        """
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
        """
        Method to print the results in a nice readable format.
        """
        if self.all_results:
            print("Model    |    Score\n--------------------\n")
            for result in self.all_results:
                print(result[2], "   |   ", result[1],"\n")
                
from itertools import product
from zwml.utilities import cross_val

class randomized_search_cv():
    
    def __init__(self, model_name, param_grid, n_iters=10, k=5):
        """
        Given a base model and a parameter grid of params
        for that model, goes through n_iters iterations and 
        builds a model with a sample from the provided params,
        and does Kfold cross validation on that model.
        ---
        Inputs:
        model_name : the name of the model with parenthesis 
        and as a string. Any parameters you wish to set for all
        models can be set in the parameter name.
        param_grid: dictionary with parameter names as keys,
        and the appropriate values for parameter. 
            Param Options:
            ('gaus', mean, std dev) # normal with mean and std dev
            ('uni', low, high) # uniform between low and high
            ('randint', low, high) # int between low and high
            ('binom', p) # binomial sample with prob p
            ('choose', list) #picks one of the list
        n_iters: how many models to build with randomized params
        k: number of folds (int)
        """
        self._base_model = str(model_name).replace(')','')
        self._param_grid = param_grid
        self.n_iters = int(n_iters)
        self.k = k
        self.models = self.get_models()  
        
    def get_models(self):
        """
        For every iteration of the model, goes parameter
        by parameter and samples the distribution provided
        for the parameter. Creates a string that is the model
        name with the parameters as part of the string.
        """
        command_list = []
        for _ in range(self.n_iters):
            cmd = self._base_model
            if cmd[-1] != '(':
                cmd+=', '
            for key, value in self._param_grid.items():
                if value[0] == "choose":
                    cmd += str(key)+"='"+str(self.sampling(value))+"', "
                else:
                    cmd += str(key)+"="+str(self.sampling(value))+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
    def sampling(self, params):
        """
        Given a parameter set in the format (type, args...),
        choose a value using numpy's distributions.
        Param Options:
            ('gaus', mean, std dev) # normal with mean and std dev
            ('uni', low, high) # uniform between low and high
            ('randint', low, high) # int between low and high
            ('binom', p) # binomial sample with prob p
            ('choose', list) #picks one of the list
        """
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
        elif dist == 'choose':
            try:
                return np.random.choice(params[1])
            except:
                raise ValueError("'choose' must have 1 parameter, the list of possibilities")
        else:
            raise ValueError("Distribution must be in ['gaus', 'uni','randint','binom','choose']")
    
    def fit(self, X, y):
        """
        Uses the "eval" function in Python to convert the model
        name from string to an actual model. Fits each model
        and scores it with kfold cross_val. 
        Creates a lists of models and scores.
        Sets the best possible model and score to be easily
        retrievable and usable.
        """
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
        """
        Method to print the results in a nice readable format.
        If the user asks for mean, only show the average score 
        across all folds. If the user asks for coefficients
        show coefficients if the model has them.
        """
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