from itertools import product

class grid_search():
    
    def __init__(self, model_name, param_grid):
        """
        Given a base model and a parameter grid of params
        for that model, iterates through all the combinations
        of parameters, builds a model with each combo,
        and returns the score of the model.
        ---
        Inputs:
        model_name : the name of the model with parenthesis 
        and as a string. Any parameters you wish to set for all
        models can be set in the parameter name.
        param_grid: dictionary with parameter names as keys,
        and list of param values to test as value for each key
        """
        self._base_model = str(model_name).replace(')','')
        self._param_grid = param_grid
        self.models = self.get_models()
        
    def get_models(self):
        """
        Finds every combination of parameters from the param grid.
        Uses the string basename for to create a list of model 
        names with the proper parameters. This command_list is
        still in string form until we're ready to test the models.
        """
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
                if type(j) == type('string'):
                    cmd += str(i)+"='"+str(j)+"', "
                else:
                    cmd += str(i)+"="+str(j)+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
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

class grid_search_cv():
    
    def __init__(self, model_name, param_grid={}, k=5):
        """
        Given a base model and a parameter grid of params
        for that model, iterates through all the combinations
        of parameters, builds a model with each combo,
        and does kFold cross validation on them model
        ---
        Inputs:
        model_name : the name of the model with parenthesis 
        and as a string. Any parameters you wish to set for all
        models can be set in the parameter name.
        param_grid: dictionary with parameter names as keys,
        and list of param values to test as value for each key
        k: number of folds for cross val
        """
        self._base_model = str(model_name).replace(')','')
        self._param_grid = param_grid
        self.models = self.get_models()
        self.k = k
        
    def get_models(self):
        """
        Finds every combination of parameters from the param grid.
        Uses the string basename for to create a list of model 
        names with the proper parameters. This command_list is
        still in string form until we're ready to test the models.
        """
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
                if type(j) == type('string'):
                    cmd += str(i)+"='"+str(j)+"', "
                else:
                    cmd += str(i)+"="+str(j)+", "
            command_list.append(cmd[:-2]+')')
        return command_list
    
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