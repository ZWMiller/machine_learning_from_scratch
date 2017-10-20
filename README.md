TO DO:

  [ ] Update Naive Bayes Modules

  [ ] Markdown KMeans Notebook

  [ ] Add KMeans README

---
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

# Notebooks/modules

## Regression: 

#### linear_regression_closed_form.ipynb (linear_regression.py)

This modules uses the Linear Algebra, closed-form solution for solving for
coefficients of linear regression. 

#### stochastic_gradient_descent_regressions.ipynb (sgd_regressor.py)

This module performs stochastic gradient descent to find the regression
coefficients for linear regression. There are a few options to set, such as
learning rate, number of iterations, etc. There's also an option for setting
the learning rate to be dynamic. 

#### decision_tree_regreeor.ipynb (decision_tree_regressor.py)

This module uses optimization of standard deviation or absolute errors to build decisions trees for
regression. It will be the basis for our random
forest regressor. It has a few setting like max-depth to control how our
trees are built and a few options for optimization method.

#### random_forest_regressor.ipynb (random_forest_regressor.py)

This is similar to the random_forest_classifier, but we instead focus on getting a continuous output.

## Classification:

#### decision_tree_classifier.ipynb (decision_tree_classifier.py)

This module uses information gain to build decisions trees for
classification. It will be the basis for our bagging classifier and random
forest classifier. It has a few setting like max-depth to control how our
trees are built.


#### k_nearest_neighbors.ipynb (knn_classifier.py)

This module is based on the wisdom of "points that are close together should
be of the same class." It measures the distances to all points and then finds
the k (user specifies 'k' by setting 'n_neighbors') closest points. Those points all get to vote on
what class the new point likely is. 

#### bagging_classifier.ipynb (bagging_classifier.py)

This ensemble method is an extension on the decision tree that uses
bootstrapping. Bootstrapping where we sample the dataset (with replacement)
over and over to build out new datasets that "built from" our true data. If we
do this many times, we'll build many slightly different trees on the bootstrapped data
since no two trees will see the exact same data. Then we let all the trees
predict on any new data, and allow the wisdom of the masses to determine our
final outcome.

#### random_forest_classifier.ipynb (random_forest_classifier.py)

This is another ensemble method. It's just like the bagging_classifier, except
we also randomize what features go to each tree in our data. Instead of just
randomizing our datapoints, we also say, "this tree only gets features 1, 3,
and 5." This further randomizes out input to each tree, helping to fight
over-fitting; which puts us in a better spot for the bias-variance trade off.

#### bernoulli_naive_bayes.ipynb (bernoulli_naive_bayes.py)

Uses Bayes rule to calculate the probability that a given observation will belong in each class, 
based on what it's learned about probability distributions in the training data. In the Bernoulli 
flavor, only "on" or "off" is counted for each feature when determining probability

#### gaussian_naive_bayes.ipynb (gaussian_naive_bayes.py)

Uses Bayes rule to calculate the probability that a given observation will belong in each class, 
based on what it's learned about probability distributions in the training data. In the Gaussian 
flavor, each feature is assumed to have a normal distribution, so the sample mean and standard deviation are used
to approximate the Probability Distribution; which is sampled to determine probability.

## Clustering:

#### KMeans

Description still to come. 

## Non-Algorithm - but useful

#### train_test_and_cross_validation.ipynb (data_splitting.py)

We use different methods of splitting the data to measure the model
performance on "unseen" or "out-of-sample" data. The cross-validation method
will report the model behavior several different folds. Both cross validation
and train-test split are built from scratch in this notebook. 

#### stats\_regress.py (no notebook, just used for quick calculations)

This is a suite of statistics calculation functions for regressions. Examples:
mean_squared_error, r2, adjusted r2, etc.


## _Methodology note:_

A lot of these modules are *begging* for inheritance. As an example, the
bagging classifier and the random forest classifier are largely the same code,
with a few modified methods. Since these are designed as pedagogical tools and
not "production code," I've chosen to make the modules as self-contained as
possible. So instead of having an abstracted parent class, which a new
programmer may have to track down, I've chosen to keep the code all together.
I know it's sub-optimal for production, but I think it's better for someone to
learn from. The only exceptions are ensemble methods that call entire other
algorithms. For instance, the random forest module is building a bunch of
decision trees, but with modfied data inputs. To illustrate this point, the
decision tree class is imported as a stand-alone module and plugged in to the
random forest module where it belongs - instead of recreating the decision
tree in that class. The idea is that a new student will see how random forest
(or other ensemble methodology) is just a super-class that wraps around
another algorithm.

