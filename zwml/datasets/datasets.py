import pandas as pd
import numpy as np
import os

def load_iris(as_dataframe=False):
    directory, _ = os.path.split(__file__)
    DATA_PATH = os.path.join(directory, "iris.data")
    data = pd.read_csv(DATA_PATH, header=None)
    data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
    if as_dataframe:
        return data
    X = data.iloc[:,:-1].as_matrix()
    y = data.iloc[:,-1]
    y = y.str.replace('Iris-setosa','0').replace('Iris-versicolor','1').replace('Iris-virginica','2')
    y = y.astype(int).as_matrix()
    return X,y