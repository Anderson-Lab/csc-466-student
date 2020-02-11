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


**Instructions:** Pair programming assignment. Submit only a single notebook, but make sure to include your first and last names.


# Bayesian Classifier

## Preface
(Courtesy of Dr. Alex Dekhtyar)

The core objective of Knowledge Discovery in Data/Data Mining/Machine Learning methods is to provide efficient algorithms for gaining insight from data. CSC 466 primarily studies the methods and the algorithms that enable
such insight, and that specifically take this insight above and beyond traditional statistical analysis of data (more
about this — later in the course).
However, the true power of KDD/DM/ML methods that we will study in this course is witnessed only when
these methods are applied to actually gain insight from the data. As such, in this course, the deliverables for your
laboratory assignments will be partitioned into two categories:

1. KDD Method implementation. In most labs you will be asked to implement from scratch one or more
KDD method for producing a special type of insight from data. This part of the labs is similar to your other
CS coursework - you will submit your code, and, sometimes, your tests and/or output.

2. Insight, a.k.a., data analysis. For each lab assignment we will provide one or more datasets for your
perusal, and will ask you to perform the analysis of these datasets using the methods you implemented. The
results of this analysis, i.e., the insight, are as important for successful completion of your assignments, as
your implementations. Most of the time, you will be asked to submit a lab report detailing your analysis,
and containing the answers to the questions you are asked to study.
The insight portion of your deliverables is something that you may be seeing for the first time in your CS
coursework. It is not an afterthought in your lab assignments. Your grade will, in no small part, depend on
the results of your analysis, and the writing quality on your report. This lab assignment, and further assignments
will include detailed insturctions on how to prepare reports, and we will discuss report writing several times as
the course progresses.

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


For this lab, we are going to first implement a empirical naive bayesian classifier, then implement a feature importance measure and apply it to a dataset, and finally, we will examine the affect of modifying the priors.

For developing this lab, we can use the Titanic Kaggle dataset.

```python
import pandas as pd
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.head()
```

We only need a few columns, and I will also perform some preprocessing for you:

```python
features = ['pclass','survived','sex','age']
titanic_df = titanic_df.loc[:,features]
display(titanic_df)
titanic_df.loc[:,'pclass']=titanic_df['pclass'].fillna(titanic_df['pclass'].mode()).astype(int)
titanic_df.loc[:,'age']=titanic_df['age'].fillna(titanic_df['age'].median())
titanic_df.loc[:,'age']=(titanic_df['age']/10).astype(str).str[0].astype(int)*10
titanic_df
```

```python
titanic_df.describe()
```

## Exercise 0
In your own words, describe the preprocessing steps I took above.




## Exercise 1
Fill in the following function to determine the prior probability of the classes. The result must be in the form of a Python dictionary such as ``priors = {0: 0.4, 1: 0.6}``.
<pre>
def compute_priors(y):
  ???
  return priors
</pre>

```python
# YOUR SOLUTION HERE
compute_priors(titanic_df['survived'],yname='survived')
```

## Exercise 2
The next function to implement is the specific class conditional probability:
<pre>
def specific_class_conditional(x,xv,y,yv):
  ???
  return prob
</pre>

```python
# YOUR SOLUTION HERE
specific_class_conditional(titanic_df['sex'],'female',titanic_df['survived'],0)
```

## Exercise 3
Now construct a dictionary based data structure that stores all possible class conditional probabilities (e.g., loop through all possible combinations of values). The keys in your dictionary should be of the form "pclass=1|survived=0".

<pre>
# X is a dataframe that does not contain the class column y.
def class_conditional(X,y):
  ???
  return probs
</pre>

```python
# YOUR SOLUTION HERE
display(class_conditional(titanic_df.drop("survived",axis=1),titanic_df["survived"]))
display(class_conditional(titanic_df.drop("survived",axis=1),titanic_df["survived"],yname="survived"))
```

## Exercise 4
Now you are ready to calculate the posterior probabilities for a given sample. Write and test the following function that returns a dictionary where the keys are of the form "survived=0|pclass=1,sex=male,age=60". Make sure you return 0 if the specific combination of values does not exist.
<pre>
def posterior(probs,priors,x):
    return posteriors
</pre>

```python
# YOUR SOLUTION HERE
probs = class_conditional(titanic_df.drop("survived",axis=1),titanic_df["survived"],yname="survived")
priors = compute_priors(titanic_df["survived"],yname="survived")
posteriors(probs,priors,titanic_df.drop("survived",axis=1).loc[0])
```

## Exercise 5
All this is great, but how would you evaluate how we are doing? Let's write a function call train_test_split that splits our dataframe into approximately training and testing dataset. Make sure it does this randomly.
<pre>
def train_test_split(X,y,test_frac=0.5):
   return Xtrain,ytrain,Xtest,ytest
</pre>

```python
# YOUR SOLUTION HERE
Xtrain,ytrain,Xtest,ytest=train_test_split(titanic_df.drop("survived",axis=1),titanic_df["survived"])
Xtrain,ytrain,Xtest,ytest
```

## Exercise 6
For this exercise, find the conditional probabilities and the priors using a training dataset of size 70% and then using these probabilities find the accuracy if they are used to predict the test dataset. 

```python
Xtrain,ytrain,Xtest,ytest=train_test_split(titanic_df.drop("survived",axis=1),titanic_df["survived"])
# YOUR SOLUTION HERE
```

## Exercise 7
For this exercise, you must improve/extend your methods above as necessary to compute the accuracy of predicting the activity from the dataset we've generated in class. Once we have filled out this dataset, I will provide a csv file as well as any preprocessing similar to the Titanic. You may have to modify your functions above to work with both datasets or you may not (depending of course on how you wrote them).

```python
# YOUR SOLUTION HERE
print("Test set accuracy:",sum(predictions==ytest)/len(ytest))
pd.DataFrame({'prediction':predictions,'activity':ytest})
```

## Excercises 8
For this exercise, I would like you to implement the feature importance algorithm describe in [https://christophm.github.io/interpretable-ml-book/feature-importance.html](https://christophm.github.io/interpretable-ml-book/feature-importance.html). After you implement this, what is the most important feature for our in-class activity prediction dataset? Does this feature make sense to you?

```python
# YOUR SOLUTION HERE
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
