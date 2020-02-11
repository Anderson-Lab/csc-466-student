---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**PUT YOUR FULL NAME(S) HERE**


## Should we grade this notebook? (Answer yes or no)


???YES OR NO???


**Instructions:** Pair programming assignment. Submit only a single notebook unless you deviate significantly after lab on Thursday. If you submit individually, make sure you indicate who you worked with originally. Make sure to include your first and last names. For those students who push to individual repos but still work in groups, please indicate which notebook should be graded.


# Ensemble Learning and Perceptron

## Lab Assignment

This is a pair programming assignment. I strongly
discourage individual work for this (and other team/pair programming) lab(s), even if you think you can do it
all by yourself. Also, this is a pair programming assignment, not a ”work in teams of two” assignment. Pair
programming requires joint work on all aspects of the project without delegating portions of the work to individual
1
team members. For this lab, I want all your work — discussion, software development, analysis of the results,
report writing — to be products of joint work.
Students enrolled in the class can pair with other students enrolled in the class. Students on the waitlist can
pair with other students on the waitlists. In the cases of ”odd person out” situations, a team of three people can
be formed, but that team must (a) ask and answer one additional question, and (b) work as a pair would, without
delegation of any work off-line.


## At the end of this lab, I should be able to
* Formulate your own questions and understand how you can go about getting answers
* Understand how to select an algorithm for your task
* Implement ensemble methods gradient boosting and random forest


## Our data
We will be using a well known housing dataset from Boston.
<pre>
 Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
</pre>

```python
import pandas as pd
df = pd.read_csv("housing/boston_fixed.csv")
df.head()
```

**Exercise 1.** Read the descriptions of the questions above, and come up with three reasonable questions with corresponding methods to test them. The only one that you cannot write, is the one we will do as a class, which I use as an example here:

Example questions: 
* What are factors that are most predictive of the median value of owner-occupied homes? 
* Is there a small subset of the total number of variables that could be used in predictive model and not sacrifice model accuracy?
* Can we say that any of these factors are causing the median home values to go up? 

Methodology:
1. Empirically determine the best modeling method from our known list of ensemble learners and decision trees.
2. Using this best model, compute a feature importance score
3. Graph the feature importance score and see if this is a dip. Use this as a cutoff if so, if not, then select the best N features and verify model performance does not change significantly.
4. NO!!! We cannot say anything about causation with our machine learning models. There are a lot of good discussions out there on why we can't say much about casuation. [See this one for example](https://towardsdatascience.com/causality-in-machine-learning-101-for-dummies-like-me-f7f161e7383e). BUT we can say a bit about correlation and what features are impacting our overall model.


**YOUR SOLUTION HERE**


**For the next few questions, we will lean heavily upon sklearn and the built-in models. We'll implement our own methods later in the lab, but this is better to provide a consistent experience.**


**Exercises 2-9**
What are the factors that are most predictive of the median value of owner-occupied homes? Use the following methodology:

1. Empirically determine the best modeling method from our known list of ensemble learners and decision trees (see code for more details)
2. Using this best model, compute a feature importance score and rank the features by this


### Code to get you started
I included all of the imports I used in this section right here. I encourage you to take a look at their documentation. I also encourage you to try and mess with the parameters yourself and see if you can come up with better combinations. Finally, you can completely break the overall flow of what I've laid out as long as you accomplish the main goals.

```python
%%capture 
!pip install scikit-posthocs
```

```python
# this is for plotting
%matplotlib inline 

import copy

# our standard imports
import numpy as np
import pandas as pd

# of course we need to be able to split into training and test
from sklearn.model_selection import train_test_split

# we need a "loss" function
from sklearn.metrics import mean_squared_error, r2_score

# This is where we can get our models
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor

# This is what I used for comparing my models
import scipy.stats as stats
import scikit_posthocs as sp

X = df.drop("MEDV",axis=1)
y = df["MEDV"]

# Below are sample arguments, manually modify some of them and see what happens (we'll do this another time with grid search)
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gb_1 = ensemble.GradientBoostingRegressor(**params)
params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gb_2 = ensemble.GradientBoostingRegressor(**params)
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
rf_1 = RandomForestRegressor(n_estimators=100)
rf_2 = RandomForestRegressor(n_estimators=500)

models = [('Gradient Boosting 1',gb_1),('Gradient Boosting 2',gb_2),
          ('DTree 1',regr_1),('DTree 2',regr_2),
          ('RF 1',rf_1),('RF 2',rf_2)
         ]
```

**Exercise 2** Fill in the following code that finds the mean squared error for 30 repeated hold-out cross-validation experiments for each classifier. In other words, fill in my code and produce something similar to my output. It is very important to realize that you will get different numbers since this is stochastic.

```python
num_iterations = 30
predictions = []
ytests = []
for iteration in range(num_iterations):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.10,shuffle=True)
    models_train = copy.deepcopy(models)
    # YOUR SOLUTION HERE
        
errors = {}
for desc,model in models_train:
    errors[desc] = []
    for iteration in range(num_iterations):
        # YOUR SOLUTION HERE
errors = pd.DataFrame(errors)
```

```python
errors.head()
```

```python
errors.describe()
```

**Exercise 3** Perform a one-way ANOVA to determine if there are any significant differences between methods

```python
```

**Exercise 4** Perform a post-hoc pairwise test with bonferroni multiple test correction

```python
```

**Exercise 5** Which method(s) perform the best? Consider which methods you can actually say with certainty perform better than the rest.




**Exercise 6** Spoiler... There should be more a few models that we are unable to distinguish using 30 trials. Rerun your above analysis, but this time repeat it with 200 trials instead of 30. Is there now a clear winner? This can definitely take a while...

```python
num_iterations = 200
predictions = []
ytests = []
for iteration in range(num_iterations):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.10,shuffle=True)
    models_train = copy.deepcopy(models)
    # YOUR SOLUTION HERE
        
errors = {}
for desc,model in models_train:
    errors[desc] = []
    for iteration in range(num_iterations):
        # YOUR SOLUTION HERE
errors = pd.DataFrame(errors)
```

```python
```

**Exercise 6** Are there still any ties? If so, what are the best models? From there select the top model in terms of average error. Would this have been your same conclusion with only 30 experiments?




**Exercise 7** With you model of choice, calculate the mean_squared_error and r2_score.

```python
```

**Exercise 8** Now compute feature importance using the method we've developed in previous labs. I have two loops here. One is that I rerun train_test_split 50 times as you can see from above this makes a difference. Then I also permute each feature 100 times. Test your code with much smaller numbers.

```python
num_iterations = 10
percent_diff_score = {}
iterations = {}
experiments = {}
for iteration in range(num_iterations):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.10,shuffle=True)
    # YOUR SOLUTION HERE
    

```

```python
# NOTE: I did (new_score-orig_score)/orig_score, so the most important feature is the one with the largest average difference
percent_diff_score_data=pd.DataFrame(percent_diff_score).describe().loc['mean'].sort_values()
display(percent_diff_score_data)
percent_diff_score_data.plot.bar()
```

Based on the analysis when this notebook was last run, I would say that the three most important features are RM, DIS, and LSTAT. Let's see what happens when we compare our performance.

```python
num_iterations = 30
predictions = []
ytests = []
for iteration in range(num_iterations):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.10,shuffle=True)
    # YOUR SOLUTION HERE
```

```python
errors = {}
for desc in ['All features','Subset']:
    errors[desc] = []
    for iteration in range(num_iterations):
        # YOUR SOLUTION HERE
errors = pd.DataFrame(errors)
errors.describe()
```

I would say that is not too bad of a difference in score considering we are only using 3 of the features.


**Exercise 9** Now what if I told you that Random Forest and other classifiers have built-in measures for feature importance. Run the following code and compare the feature importance scores. The calculation of these needs to be saved for another time and place, but the trees themselves contain information about feature importance based on the location in the tree a feature is most often selected.

```python
forest = RandomForestRegressor(n_estimators=500)
forest.fit(X,y)
importances = forest.feature_importances_
importances = pd.DataFrame({'Feature':X.columns,'Importance':importances})
importances.sort_values(by='Importance').set_index('Feature').plot.bar()
```



**Implementation from scratch portion**: We are now going to implement two ensemble learning methods from scratch and see how our implementations compare to sklearn.


**Exercise 10** Implement a simple random forest classifier and compare the performance to one of the random forest classifiers above.

```python
```

```python
errors = {}
for desc in ['sklearn RF','Our RF']:
    errors[desc] = []
    for iteration in range(num_iterations):
        # YOUR SOLUTION HERE
errors = pd.DataFrame(errors)
errors.describe()
```

**Exercise 10** Implement gradient boosting from scratch using a mean squared error loss function. Compare the performance. I "boosted" 100 times. I've shown my validation graph. Every run is a little different, and it would definitely make this algorithm smarter if you stopped based on the validation graph.

```python
```

```python
pd.DataFrame({"Validation MSE":val_mses}).plot.line()
```

```python
errors = {}
for desc in ['sklearn GB','Our Gradient Boosting']:
    errors[desc] = []
    for iteration in range(num_iterations):
        # YOUR SOLUTION HERE
errors = pd.DataFrame(errors)
errors.describe()
```
