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

#### decision_tree_scratch.ipynb

This module uses information gain to build decisions trees for classification. It also has the
capability of act as Random Forest Classifier as well as a bagging classifier
if the user sets the appropriate flags.
