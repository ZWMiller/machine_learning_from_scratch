from itertools import product

class grid_search():
    
    def __init__(self, model_name, param_grid):
        self._base_model = str(model_name).replace(')','')
        self._param_grid = param_grid
        self.models = self.get_models()
        
    def get_models(self):
        params = []
        order = []
        for key, value in self._param_grid.items():
            order.append(key)
            params.append(value)
        options = list(product(*params))

        command_list = []
        for option in options:
            cmd = self._base_model
            if cmd[-1] != '(':
                cmd+=', '
            for i,j in zip(order, option):
                cmd += str(i)+"="+str(j)+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
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

class grid_search_cv():
    
    def __init__(self, model_name, param_grid={}, k=5):
        self._base_model = str(model_name).replace(')','')
        self._param_grid = param_grid
        self.models = self.get_models()
        self.k = k
        
    def get_models(self):
        params = []
        order = []
        for key, value in self._param_grid.items():
            order.append(key)
            params.append(value)
        options = list(product(*params))

        command_list = []
        for option in options:
            cmd = self._base_model
            if cmd[-1] != '(':
                cmd+=', '
            for i,j in zip(order, option):
                cmd += str(i)+"="+str(j)+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
    def fit(self, X, y):
        results = []
        for model_name in self.models:
            model = eval(model_name)
            cv = cross_val()
            cv.cross_validation_scores(model, X, y, self.k)
            results.append([model, cv.score_folds, model_name])
        self.all_results = sorted(results, key=lambda x: x[1], reverse=True)
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