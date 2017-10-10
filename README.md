# Machine Learning from Scratch in Python


### If you want to understand something, you have to be able to build it. 

This is my attempt to build many of the machine learning algorithms from
scratch, both in an attempt to make sense of them for myself and to write the
algorithms in a way that is pedagogically interesting. At present, SkLearn is
the leading Machine Learning module for Python, but looking through the
open-source code, it's very hard to make sense of because of how abstracted
the code is. These modules will be much simpler in design, such that a student
can read through and understand how the algorithm works. As such, they will
not be as optimized as SkLearn, etc.

**__Organization__**

Notebooks: Each notebook will have the class fully written out, with a test case shown.
All version information for the used python and modules (numpy, pandas, etc)
are shown as well for later comparison. 

Modules: These files will simply contain the class/functions as an importable
module for use with outside data.

## Notebooks/modules


#### linear_regression_closed_form.ipynb

This modules uses the Linear Algebra, closed-form solution for solving for
coefficients of linear regression. 

#### stochastic_gradient_descent_regressions.ipynb

This module performs stochastic gradient descent to find the regression
coefficients for linear regression. There are a few options to set, such as
learning rate, number of iterations, etc. There's also an option for setting
the learning rate to be dynamic. 

#### stats\_regress.py

This is a suite of statistics calculation functions for regressions. Examples:
mean_squared_error, r2, adjusted r2, etc.

#### decision_tree_classifier.ipynb

This module uses information gain to build decisions trees for
classification. It will be the basis for our bagging classifier and random
forest classifier. It has a few setting like max-depth to control how our
trees are built.

#### k_nearest_neighbors.ipynb

This module is based on the wisdom of "points that are close together should
be of the same class." It measures the distances to all points and then finds
the k (user specifies 'k' by setting 'n_neighbors') closest points. Those points all get to vote on
what class the new point likely is. 

#### train_test_and_cross_validation.ipynb (data_splitting.py)

We use different methods of splitting the data to measure the model
performance on "unseen" or "out-of-sample" data. The cross-validation method
will report the model behavior several different folds. Both cross validation
and train-test split are built from scratch in this notebook. 
