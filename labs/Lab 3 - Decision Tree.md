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


**Instructions:** Pair programming assignment. Submit only a single notebook unless you deviate significantly after lab on Thursday. If you submit individually, make sure you indicate who you worked with originally. Make sure to include your first and last names.


# Decision Trees

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


For this lab, we are going to implement a variation of the C4.5 decision tree with statistical pruning. C4.5 provides several improvements over ID3 though the base structure is very similar. Note that there are different methods for pruning a decision tree. Here we will use statistical confidence intervals. If you have trouble getting started, don't forget Marsland has sample code on his website that corresponds to the book.

We will start with our titanic dataset first.

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

## Exercise 1
Construct a function that calculates the entropy of a set (Pandas Series Object).
<pre>
def entropy(y):
  ???
  return e
</pre>

```python
# YOUR SOLUTION HERE
display(entropy(titanic_df['survived']))
display(entropy(pd.Series([0,0,0,1,1,1])))
display(entropy(pd.Series([0,0,0])))
```

## Exercise 2
Now write a function that calculates the information gain after splitting with a specific variable (Equation 12.2 from Marsland).
<pre>
def gain(y,x):
  ???
  return g
</pre>

```python
# YOUR SOLUTION HERE
display(gain(titanic_df['survived'],titanic_df['sex']))
display(gain(titanic_df['survived'],titanic_df['pclass']))
display(gain(titanic_df['survived'],titanic_df['age']))
```

## Exercise 3
C4.5 actually uses the gain ratio which is defined as the information gain "normalized" (divided) by the entropy before the split. You have written everything you need here. Just put it together.

<pre>
def gain_ratio(y,x):
  ???
  return gr
</pre>

```python
# YOUR SOLUTION HERE
display(gain_ratio(titanic_df['survived'],titanic_df['sex']))
display(gain_ratio(titanic_df['survived'],titanic_df['pclass']))
display(gain_ratio(titanic_df['survived'],titanic_df['age']))
```

## Exercise 4
Define a function that chooses the column to split the tree out of possible columns in the dataframe.

<pre>
def select_split(X,y):
   ???
   return column,gain
</pre>

```python
# YOUR SOLUTION HERE
select_split(titanic_df.drop('survived',axis=1),titanic_df['survived'])
```

## Exercise 5
Now put it all together and construct a function called make_tree that returns a tree in a proprietary format of your choosing. Mine is a dictionary, but do whatever you want as long as it works :)

HINT: Don't forget to look at the base case first.
<pre>
def make_tree(X,y):
   ???
   return tree
</pre>

```python
# YOUR SOLUTION HERE
tree = make_tree(titanic_df.drop('survived',axis=1),titanic_df['survived'])
display(tree)

# if you want to print like me :)
def print_for_me(tree):
    import json
    import copy
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))
print_for_me(tree)
```

## Exercise 6
Modify your code above to deal with age as continuous column and include a new parameter to make tree that is the minimum number of samples required in order to consider splitting (default value should be 5). My solution was to redefine select_split to check the types. Pandas has a built in categorical object type, so it is wise to make use of it and modify our dataset as follows:

```python
titanic_df.dtypes
```

```python
titanic_df['pclass'] = titanic_df['pclass'].astype('category')
titanic_df['survived'] = titanic_df['survived'].astype('category')
titanic_df['sex'] = titanic_df['sex'].astype('category')
titanic_df.dtypes
```

```python
# YOUR SOLUTION HERE
tree = make_tree(titanic_df.drop('survived',axis=1),titanic_df['survived'])
display(tree)

# if you want to print like me :)
def print_for_me(tree):
    import json
    import copy
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
            else:
                return tree
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))
print_for_me(tree)
```

## Exercise 7
Now implement a version of pruning that uses confidence intervals of the accuracy. First, let's calculate the 90% (z=1.64) confidence intervals for a node:
<pre>
def confidence_interval(y):
    ???
    return lower,upper
</pre>

Here is the formula I want you to use:
<pre>
c.i. = f +- z*sqrt( f*(1-f) / N )
</pre>
where f is the fraction of errors (1-accuracy) and N is the number of samples.

```python
# YOUR SOLUTION HERE
display(confidence_interval(titanic_df['survived']))
display(confidence_interval(pd.Series([0,0,0,1,1,1])))
display(confidence_interval(pd.Series([0,0,0])))
```

## Excercises 8
Now calculate the conditional confidence interval (very similar in structure to conditional entropy).

<pre>
def conditional_confidence_interval(preconditions,X,y):
    ???
    return lower,upper,ypred
</pre>

```python
# YOUR SOLUTION HERE
display(conditional_confidence_interval([('sex', 'male'), ('age<75.00', 'True'), ('pclass', 2)],
                                        titanic_df.drop('survived',axis=1),titanic_df['survived']))

```

## Excercises 9
Now we can put together a pruning algorithm. Please implement statistical pruning, but here a short description of reduced-error pruning as well. Original source of some text: [http://www.cs.bc.edu/~alvarez/ML/statPruning.html](http://www.cs.bc.edu/~alvarez/ML/statPruning.html)

**Reduced-Error Pruning**: One approach to pruning is to withhold a portion of the available labeled data for validation. The validation set is not used during training. Once training has been completed, testing is carried out over the validation set. If the error rate of the original decision tree over the validation set exceeds the error rate of a pruned version of the tree (obtained by replacing a near-leaf node with its child leaves by a single leaf node), then the pruning operation is carried out. Reduced error pruning can reduce overfitting, but it also reduces the amount of data available for training.

**Statistical Pruning**: C4.5 uses a pruning technique based on statistical confidence estimates. This technique has the advantage that it allows all of the available labeled data to be used for training.

The heart of the statistical pruning technique is the calculation of a confidence interval for the error rate. In brief, one starts from an observed error rate f measured over the set of N training instances. In order to decide whether to replace a near-leaf node and its child leaves by a single leaf node, C4.5 compares the upper limits of the error confidence intervals for the two trees. For the unpruned tree, the upper error estimate is calculated as a weighted average over its child leaves (what you implemented above). Whichever tree has a lower estimated upper limit on the error rate "wins" and is selected.

<pre>
def generate_rules(tree,rules=[]):
   ???
   
def prune_rules(rules,X,y):
   return new_rules # (sorted by accuracy)
</pre>

```python
# YOUR SOLUTION HERE
tree = make_tree(titanic_df.drop('survived',axis=1),titanic_df['survived'])
rules = generate_rules(tree)
print('Original rules:')
for rule in rules:
    print(rule)
    
new_rules = prune_rules(rules,titanic_df.drop('survived',axis=1),titanic_df['survived'],debug=True)
print('I return a dataframe so I can sort the values')
print(new_rules)
print('Here they are sorted')
print(new_rules.sort_values(by='upper'))
print('Now put them back into a list in this correct order')
new_rules = new_rules.sort_values(by='upper')['rule'].tolist()
for rule in new_rules:
    print(rule)
```

# Exercise 10
Now let's run our algorithm on our activity dataset. What is the resulting tree before and after pruning? Was anything pruned? What are the rules after pruning?

```python
# YOUR SOLUTION HERE
```

## Exercise 11
Considering the titanic dataset, what is the most important feature? How does the test set accuracy compare to the accuracy using naive Bayes from last lab? Does pruning help this accuracy?

HINT: It makes sense to run the combination of train+test at least 20 times to make sure we aren't just getting lucky with a single run. 

```python
# YOUR SOLUTION HERE


```

## Exercise 12
So pruned helps somewhat in this example, but we are mostly saved the need to pruning because we are only considering three variables and we are only considering age once in the tree. Not much chance of overfitting. What happens if we include the additional columns of fare, embarked, cabin, home.dest, parch, and sibsp?

HINT: It makes sense to run the combination of train+test at least 20 times to make sure we aren't just getting lucky with a single run. 

```python
import pandas as pd
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.head()
features = ['pclass','survived','sex','age','fare','embarked','cabin','home.dest','parch','sibsp']
titanic_df = titanic_df.loc[:,features]
titanic_df.loc[:,'survived']=titanic_df['survived'].astype('category')
titanic_df.loc[:,'pclass']=titanic_df['pclass'].fillna(titanic_df['pclass'].mode().values[0]).astype('category')
titanic_df.loc[:,'age']=titanic_df['age'].fillna(titanic_df['age'].median())
titanic_df.loc[:,'fare']=titanic_df['fare'].fillna(titanic_df['fare'].median())
titanic_df.loc[:,'embarked']=titanic_df['embarked'].fillna(titanic_df['embarked'].mode().values[0]).astype('category')
titanic_df.loc[:,'cabin']=titanic_df['cabin'].fillna(titanic_df['cabin'].mode().values[0]).astype('category')
titanic_df.loc[:,'home.dest']=titanic_df['home.dest'].fillna(titanic_df['home.dest'].mode().values[0]).astype('category')
titanic_df.loc[:,'parch']=titanic_df['parch'].fillna(titanic_df['parch'].mode().values[0]).astype('category')
titanic_df.loc[:,'sibsp']=titanic_df['sibsp'].fillna(titanic_df['sibsp'].mode().values[0]).astype('category')
titanic_df.loc[:,'sex']=titanic_df['sex'].fillna(titanic_df['sex'].mode().values[0]).astype('category')
display(titanic_df.dtypes)
display(titanic_df)
```

```python
# YOUR SOLUTION HERE

```

## Exercise 13

What modifications can you make to the pruning algorithm such that it trims more often? Did this improve the results?

```python
# YOUR SOLUTION HERE
```

## Exercise 14

For your final exercise, I would like you to write and apply two functions to both the activity dataset and the titanic dataset. Try to create a nice output in a table format that summarizes how each of the three scores vary for each dataset and for each possible value of pos_label.

<pre>
def precision(y,ypred,pos_label):
   ???
   return p
   
def recall(y,ypred,pos_label):
   ???
   return r
   
def f1(y,ypred,pos_label):
   ???
   return f
</pre>

```python
# YOUR SOLUTION HERE
```

```python

```
