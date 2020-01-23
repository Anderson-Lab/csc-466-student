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
**PUT YOUR FULL NAME(S) HERE** Shreya Tumma, Andrew Keshishian


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
import numpy as np
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
# CODE FOR TESTING
def compute_priors(a, yname="y"):
    priors = {}
    for i in a.unique():
        priors[str(a.name) + "=" + str(i)] = np.sum(np.where(np.array(a) == i, 1,0))/len(a)
    return priors

compute_priors(titanic_df["survived"])
```

## Exercise 2
The next function to implement is the specific class conditional probability:
<pre>
def specific_class_conditional(y,yv,x,xv):
  ???
  return prob
</pre>

```python
def specific_class_conditional(y, yv, x, xv):
    p_y = np.sum(np.where(np.array(y) == yv, 1, 0)) / len(y)
    x, y = np.array(x), np.array(y)
    count  = sum([1 for i in range(len(x)) if x[i] == xv and y[i] ==yv])
    p_xy = count / len(x)
    return p_xy / p_y

specific_class_conditional(titanic_df['survived'],0,titanic_df['sex'],'female')

```

## Exercise 3
Now construct a dictionary based data structure that stores all possible class conditional probabilities (e.g., loop through all possible combinations of values). The keys in your dictionary should be of the form "pclass=1|survived=0".

<pre>
# X is a dataframe that does not contain the class column y.
def class_conditional(X,y):
  ???
  return probs
  
def prior(y):
  ???
  return probs
</pre>

```python
def prior(y, yname = "y"):
    yv = y.unique()
    p_yv = {j:(np.sum(np.where(np.array(y) == j, 1, 0)) / len(y)) for j in yv}
    return p_yv

    
def class_conditional(X, y, yname="y"):
    name = str(y.name)
    v = y
    probs  = {}
    priors = prior(y)
    y = pd.DataFrame(y)
    big_frame = pd.concat([X, y], axis=1)
    for col in X.columns:
        for col_condition in X[col].unique():
            for key in compute_priors(v).keys():
                key_condition = key.split("=")[1]
                if key_condition.isdigit():
                    key_condition = int(key_condition)
                prob = len(big_frame[(big_frame[col] == col_condition) & (big_frame[name] == key_condition)]) / len(X)
                probs[str(col) + "=" + str(col_condition) + "|" + str(key)] = prob / priors[key_condition]
    return probs


display(class_conditional(titanic_df.drop("survived",axis=1),titanic_df["survived"]))
display(class_conditional(titanic_df.drop("survived",axis=1),titanic_df["survived"],yname="survived"))

```

## Exercise 4
Now you are ready to calculate the posterior probabilities for a given sample. Write and test the following function that returns a dictionary where the keys are of the form "pclass=1,sex=male,age=60|survived=0". Make sure you return 0 if the specific combination of values does not exist.
<pre>
def posteriors(probs,priors,x):
    return probs
</pre>

```python
# YOUR SOLUTION HERE
def posteriors(probs,priors, x):
    
    in_df = pd.DataFrame(x)
    p_list = []
    for i in range(len(x)):
        temp = str(in_df.index[i])+"="+str(x[i])
        a_list = []
        for key in probs.keys():
            num = 1
            k_list = key.split('|')
            
            if temp == k_list[0]:
                a_list.append((k_list[1],probs[key]))        
        p_list.append(a_list)      
    prob_dct = {}
    for m in p_list:
        for n in m:
            if n[0] not in prob_dct.keys():
                prob_dct[n[0]] = n[1]
            else:
                prob_dct[n[0]] *= n[1]
    den = 0
    for key in prob_dct.keys():
        den  += prob_dct[key] * priors[key]
    final_probs = {}
    for key in prob_dct.keys():
        name = str(key) + "|"
        for i in range(len(x)):
            if i < len(x) - 1:
                name += str(in_df.index[i]) + "=" + str(x[i]) + ","
            else:
                name += str(in_df.index[i]) + "=" + str(x[i])
        final_probs[name] = prob_dct[key] * priors[key] / den
    return final_probs
        
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
#Xtrain,ytrain,Xtest,ytest=train_test_split(titanic_df.drop("survived",axis=1),titanic_df["survived"])
#Xtrain,ytrain,Xtest,ytest

def train_test_split(X, y, test_frac=0.5):
    Xtrain = X.sample(frac=test_frac)
    ytrain = pd.DataFrame(index = Xtrain.index, columns = range(0, 1))
    ytrain.columns = [str(y.name)]
    probs = class_conditional(Xtrain,y)
    priors = compute_priors(y)
    for index, row in Xtrain.iterrows():
        check = posteriors(probs,priors, Xtrain.loc[index])
        key_max = max(check, key=check.get) 
        if check != None:
            key_list = key_max.split("|")
            k = key_list[0].split("=")
            if k[len(k) - 1].isdigit():
                k[len(k) - 1] = int(k[len(k) - 1])
            ytrain.loc[index, [str(y.name)]] = k[len(k) - 1]
    Xtest = X.drop(Xtrain.index)
    ytest = pd.DataFrame(index = Xtest.index, columns = range(0,1))
    ytest.columns = [str(y.name)]
    for index, row in Xtest.iterrows():
        ytest.loc[index, [str(y.name)]] = y[index]
    return Xtrain,ytrain,Xtest,ytest
    
Xtrain,ytrain,Xtest,ytest = train_test_split(titanic_df.drop("survived",axis=1),titanic_df["survived"])
Xtrain,ytrain,Xtest,ytest   
```

## Exercise 6
For this exercise, find the conditional probabilities and the priors using a training dataset of size 70% and then using these probabilities find the accuracy if they are used to predict the test dataset. 

```python

Xtest
```

```python
# YOUR SOLUTION HERE
Xtrain,ytrain,Xtest,ytest=train_test_split(titanic_df.drop("survived",axis=1),titanic_df["survived"], 0.7)
#Need to give ytrain and ytest a column name so the there is no key error in class conditional
def find_accuracy(Xtrain, Xtest, ytrain, ytest):
    y_attempt = pd.DataFrame(index = Xtest.index, columns = range(0,1))
    a = ytrain.squeeze()
    y_attempt.columns = [str(a.name)]
    probs = class_conditional(Xtrain, a)
    priors = compute_priors(a)
    for index, row in Xtest.iterrows():
        check = posteriors(probs,priors, Xtest.loc[index])
        key_max = max(check, key=check.get) 
        if check != None:
            key_list = key_max.split("|")
            k = key_list[0].split("=")
            if k[len(k) - 1].isdigit():
                k[len(k) - 1] = int(k[len(k) - 1])
            y_attempt.loc[index, [str(a.name)]] = k[len(k) - 1]
    comparison_df = pd.concat([ytest, y_attempt], axis=1, sort=False)
    comparison_df.columns = ["ytest", "y_attempt"]
    accuracy = len(comparison_df[comparison_df["ytest"] == comparison_df["y_attempt"]]) / len(ytest)
    return "Test set accuracy is: " + str(accuracy)

find_accuracy(Xtrain, Xtest, ytrain, ytest)
```

## Exercise 7
For this exercise, you must improve/extend your methods above as necessary to compute the accuracy of predicting the activity from the dataset we've generated in class. Once we have filled out this dataset, I will provide a csv file as well as any preprocessing similar to the Titanic. You may have to modify your functions above to work with both datasets or you may not (depending of course on how you wrote them).

```python
# YOUR SOLUTION HERE
activity_df=pd.read_csv('activity.csv')
Xtrain,ytrain,Xtest,ytest = train_test_split(activity_df.drop("deadline",axis=1),activity_df["deadline"])
find_accuracy(Xtrain,Xtest,ytrain, ytest)

```

## Excercises 8
For this exercise, I would like you to implement the feature importance algorithm describe in [https://christophm.github.io/interpretable-ml-book/feature-importance.html](https://christophm.github.io/interpretable-ml-book/feature-importance.html). After you implement this, what is the most important feature for our in-class activity prediction dataset? Does this feature make sense to you?

```python
# YOUR SOLUTION HERE
cols = activity_df.columns

def best_feature(df):
    accuracies = {}
    for col in cols:
        Xtrain,ytrain,Xtest,ytest = train_test_split(df.drop(col,axis=1),df[col])
        accuracy = float(find_accuracy(Xtrain,Xtest,ytrain, ytest).split(": ")[1])
        accuracies[col] = accuracy
    return "The feature of greatest importance is: " + str(max(accuracies, key=accuracies.get))
    
best_feature(activity_df)
```

This makes sense however because our dataset is so small, the results are subject to fluctuation

```python

```
