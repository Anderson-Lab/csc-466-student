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


# Recommendation Systems

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
* Understand how item-item and user-user collaborative filtering perform recommendations
* Explain a experiment where we tested item-item versus user-user

```python
# We need a better version
!pip install -U scikit-learn
```

## Our data
We will be using a well known movielens dataset (small version).


### Here are all the imports that I've used

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
```

```python
ratings = pd.read_csv('~/csc-466-student/data/movielens-small/ratings.csv') # you might need to change this path
```

```python
ratings = ratings.dropna()
ratings
```

```python
len(ratings.userId.unique())
```

```python
movies = pd.read_csv('~/csc-466-student/data/movielens-small/movies.csv')
```

```python
movies = movies.dropna()
movies
```

### Joining the data together
We need to join those two source dataframes into a single one called data. I do this by setting the index to movieId and then specifying an ``inner`` join which means that the movie has to exist on both sides of the join. Then I reset the index so that I can later set the multi-index of userId and movieId. The results of this are displayed below. Pandas is awesome, but it takes some getting used to how everything works.

```python
data = movies.set_index('movieId').join(ratings.set_index('movieId'),how='inner').reset_index()
#data["movieId"] = data["title"]+" "+data["movieId"].astype(str)
data = data.set_index(['userId','movieId'])[["rating"]]
data
```

### Turning data into a matrix instead of a series
The functions ``stack()`` and ``unstack()`` are called multiple times in this lab. They allow me to easily change from a dataframe to a series and back again. Below I'm changing from the Series object to a DataFrame. The important thing to note is that each row is now a user! NaN values are inserted where a user did not rate movie.

```python
data=data.unstack()
data
```

```python
data.columns
```

```python
# So it seems like having this extra 'rating' string is confusing folks a bit
# I'm going to just change it, but leave it as a tuple because it is important to realize that tuples can be indices and columns
data.columns = pd.MultiIndex.from_tuples([('movie',v[1]) for v in data.columns])
data.columns
```

## Let's take a look at some useful code together before the exercises.

First let's look at code that centers the data (important for cosine distance) and then fills in missing values as 0.

```python
data.shape
```

```python
data.mean().shape
```

```python
data_centered = (data.T-data.mean(axis=1)).T
data_centered = data_centered.fillna(0)
data_centered
```

### Now what if we want to grab a specific user? Let's say we want the one with user ID of 1.

```python
x = data_centered.loc[1]
x
```

### Finding neighborhood.
If we are hoping to predict movies for this user, then user-user collaborative filtering says find the ``N`` users that are similar. We should definitely drop out user 1 because it makes no sense to recommend to yourself. We then compute the cosine similarity between this user ``x`` and all other users in the db. We then reverse sort them, and then display the results.

```python
db = data_centered.drop(1)
sims = db.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
sorted_sims = sims.sort_values()[::-1]
sorted_sims
```

### Grabing similar users
Let's set the network size to 10, and then grab those users :)

```python
N=10
userIds = sorted_sims.iloc[:N].index
data_centered.loc[userIds]
```

### How about a prediction?
We could compute the mean from the neighborhood for each prediction

```python
db.loc[userIds].mean()+data.loc[1].mean()
```

### What if we want to weight by the distance?

```python
db.loc[userIds].multiply(sorted_sims.iloc[:N],axis=0).sum()/sorted_sims.iloc[:N].sum()+data.loc[1].mean()
```

## User-user small dataset example

```python
# grab some movies that were watched a lot
r=(data > 0).sum()
our_movies = r.sort_values(ascending=False).iloc[:10].index
our_movies
```

```python
our_data = data[our_movies] # grab only those movies
```

```python
# Now grab just the users
our_users = (our_data>0).sum(axis=1).sort_values(ascending=False).iloc[:10].index
```

```python
test_data = our_data.loc[our_users]
test_data
```

It doesn't serve our purpose to have no missing values, so let's put some back in.

```python
test_data.iloc[0,8] = np.NaN
test_data.iloc[1,8] = np.NaN
test_data.iloc[0,6] = np.NaN
test_data.iloc[5,8] = np.NaN
test_data.iloc[0,2] = np.NaN
test_data.iloc[3,8] = np.NaN
test_data.loc[274,('movie',593)] = np.NaN
test_data.loc[274,('movie',527)] = np.NaN
```

```python
test_data
```

```python
test_data = (test_data.T-test_data.T.mean()).T # mean center everything
test_data.loc[610].mean() # check the mean of user 610
```

```python
x_raw = test_data.loc[610] # x_raw is a user
x_raw
```

```python
data_raw = test_data # keep a copy of test_data that doesn't have any missing values
test_data = test_data.fillna(0) # fill in missing values
```

```python
# we need to split this up into training and test sets
train_movies, test_movies = train_test_split(x_raw.dropna(),test_size=0.2,random_state=1)
display(train_movies)
display(test_movies)
```

```python
# but we just wanted the movies and not the ratings
train_movies, test_movies = train_test_split(x_raw.dropna().index,test_size=0.2,random_state=1)
print('Training movies')
display(train_movies)
print('Testing movies')
display(test_movies)
```

```python
test_data
```

```python
db = test_data.drop(x_raw.name) # remove this user
db
```

```python
movie = ('movie',527) # pick a movie in our test set
display(db)
# We should remove any users that did not rate the movie we are interested in predicting. How would including them help us?
db_subset = db.loc[np.isnan(data_raw.drop(x_raw.name)[movie])==False]
display(db_subset)
```

```python
# In order to make the cosine similarity work, we need to have the same dimensions in db_subset and x
# But we want to make sure that the test movies are removed because well they are for testing purposes
x = x_raw.copy()
x.loc[test_movies] = np.NaN
x = x.fillna(0)
x
```

```python
# Now we can actually compute the cosine similarity. This apply function is basically just a for loop over each user
sims = db_subset.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
```

```python
N = 2 # Set the neighborhood to 2 and select the users after sorting
sims.sort_values(ascending=False).iloc[:N]
```

```python
# But we don't want the similarity scores, just the user ids
neighbors = sims.sort_values(ascending=False).iloc[:N].index
neighbors
```

```python
# How did our neighborhood rank that movie?
test_data.loc[neighbors,movie]
```

```python
# Finally! Here is our prediction (unweighted)
pred = test_data.loc[neighbors,movie].mean()
pred
```

```python
# What about weighted?
top_sims = sims.sort_values(ascending=False).iloc[:N]
top_sims
```

```python
# Here is our prediction with weighting
weighted_pred = test_data.loc[neighbors,movie].multiply(top_sims,axis=0).sum()/top_sims.sum()
weighted_pred
```

```python
# How does this compare?
actual = x_raw.loc[movie]
actual
```


```python
print("MAE of unweighted:",np.abs(actual-pred))
print("MAE of weighted:",np.abs(actual-weighted_pred))
```

## Item-item on the same small dataset
Let's review what we have from above that becomes our input

```python
data_raw
```

```python
# We are going to need to transform this
data_raw.T
```

```python
x_raw
```

```python
train_movies
```

```python
test_movies
```

```python
# This is the movie we are still trying to predict (i.e., from the testing set we pick the first one)
movie
```

The intuition behind item-item is we want to predict the rating of a movie based on user 610's ratings on similar movies. In other words, if we knew that most similar movies to 527 were 356, 318, and 296, then we would calculate our prediction like this:

```python
# the use of 'rating' is just an artifact of pandas transformations
ids = [('movie',356),('movie',318),('movie',296)]
x_raw.loc[ids]
```

```python
# so we could predict like this
x_raw.loc[ids].mean()
```

```python
# but wait, why would we even includ movie 296? The above mean ignores this in the calculation,
# so it is better to just prevent this from happening, we can do that when we search the neighborhood!
```

```python
test_data = data_raw.T.fillna(0)
test_data
```

```python
# The following three lines are the same as the single line left below
x = test_data.loc[movie].drop(x_raw.name)
x
```

```python
db_subset = test_data.loc[train_movies].drop(x_raw.name,axis=1)
db_subset
```

```python
sims = db_subset.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
sims
```

```python
top_sims = sims.sort_values(ascending=False).iloc[:N]
top_sims
```

```python
ids = top_sims.index
ids
```

```python
pred = x_raw.loc[ids].mean()
pred
```

```python
weighted_pred = x_raw.loc[ids].multiply(top_sims,axis=0).sum()/top_sims.sum()
weighted_pred
```

```python
print("MAE of unweighted:",np.abs(actual-pred))
print("MAE of weighted:",np.abs(actual-weighted_pred))
```

## Finally to the exercises!
I want you to implement user-user, item-item, and a combination of item-item and user-user.


## Exercise 1 (Worth 5 points)
Complete the following function that predicts using user-user collaborative filtering. 

```python
def predict_user_user(data_raw,x_raw,N=10,frac=0.02):
    # data_raw is our uncentered data matrix. We want to make sure we drop the name of the user we
    # are predicting:
    db = data_raw.drop(x_raw.name)
    # We of course want to center and fill in missing values
    db = (db.T-db.T.mean()).fillna(0).T
    # Now this is a little tricky to think about, but we want to create a train test split of the movies
    # that user x_raw.name has rated. We need some of them but want some of them removed for testing.
    # This is where the frac parameter is used. I want you to think about how to select movies for training
    #ix_raw, ix_raw_test = train_test_split(???,test_size=frac,random_state=42) # Got to ignore some movies
    
    # Here is where we use what you figured out above
    x_raw_test = x_raw.loc[ix_raw_test] 
    x_raw = x_raw.copy()
    x_raw.loc[ix_raw_test] = np.NaN # ignore the movies in test
    x = (x_raw - x_raw.mean()).fillna(0)

    preds = []
    for movie in ix_raw_test:
        #sims = db.loc[??? Only look at users who have rated this movie ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        try:
            sorted_sims = sims.sort_values()[::-1]
        except:
            preds.append(0) # means there is no one that also rated this movie amongst all other users
            continue
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? using ids how do you predict ???)
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test-x_raw.mean()
    mae = (actual-pred).abs().mean()
    return mae
```

```python
mae = predict_user_user(data,data.loc[1])
mae
```

```python
maes = data.head(20).apply(lambda x: predict_user_user(data,x),axis=1)
```

```python
np.mean(maes)
```

## Exercise 2 (Worth 5 points)
Complete the following function that predicts using item-item collaborative filtering. 

```python
def predict_item_item(data_raw,x_raw,N=10,frac=0.02,debug={}):
    ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42) # Got to ignore some movies
    x_raw_test = x_raw.loc[ix_raw_test]
    
    db = data_raw.drop(x_raw.name)
    db = (db.T-db.T.mean()).fillna(0).T
    # ??? db = FIX DB SO WE CAN KEEP CODE SIMILAR BUT DO ITEM-ITEM ???
    preds = []
    for movie in ix_raw_test:
        x = db.loc[movie]
        # sims = db.drop(movie).loc[??? ONLY SELECT MOVIES IN TRAINING SET WHICH USER HAS RATED ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        sorted_sims = sims.sort_values()[::-1]
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? HOW TO PREDICTION ???)
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test
    mae = (actual-pred).abs().mean()
    return mae
```

```python
mae = predict_item_item(data,data.loc[1])
mae
```

```python
maes = data.head(20).apply(lambda x: predict_item_item(data,x),axis=1)
```

```python
np.mean(maes)
```

**For this very simple experiment, what method seems better?**

YOUR ANSWER HERE


## Exercise 3 (Worth 5 points)
Create new versions of predict_user_user and predict_item_item, but now perform a weighted prediction as was demonstrated above. Did our accuracy get any better?

```python
def predict_item_item(data_raw,x_raw,N=10,frac=0.02,debug={}):
    ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42) # Got to ignore some movies
    x_raw_test = x_raw.loc[ix_raw_test]
    
    db = data_raw.drop(x_raw.name)
    db = (db.T-db.T.mean()).fillna(0).T
    # ??? db = FIX DB SO WE CAN KEEP CODE SIMILAR BUT DO ITEM-ITEM ???
    preds = []
    for movie in ix_raw_test:
        x = db.loc[movie]
        # sims = db.drop(movie).loc[??? ONLY SELECT MOVIES IN TRAINING SET WHICH USER HAS RATED ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        sorted_sims = sims.sort_values()[::-1]
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? HOW TO PREDICTION ???)
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test
    mae = (actual-pred).abs().mean()
    return mae

def predict_user_user(data_raw,x_raw,N=10,frac=0.02):
    # data_raw is our uncentered data matrix. We want to make sure we drop the name of the user we
    # are predicting:
    db = data_raw.drop(x_raw.name)
    # We of course want to center and fill in missing values
    db = (db.T-db.T.mean()).fillna(0).T
    # Now this is a little tricky to think about, but we want to create a train test split of the movies
    # that user x_raw.name has rated. We need some of them but want some of them removed for testing.
    # This is where the frac parameter is used. I want you to think about how to select movies for training
    #ix_raw, ix_raw_test = train_test_split(???,test_size=frac,random_state=42) # Got to ignore some movies
    
    # Here is where we use what you figured out above
    x_raw_test = x_raw.loc[ix_raw_test] 
    x_raw = x_raw.copy()
    x_raw.loc[ix_raw_test] = np.NaN # ignore the movies in test
    x = (x_raw - x_raw.mean()).fillna(0)

    preds = []
    for movie in ix_raw_test:
        #sims = db.loc[??? Only look at users who have rated this movie ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        try:
            sorted_sims = sims.sort_values()[::-1]
        except:
            preds.append(0) # means there is no one that also rated this movie amongst all other users
            continue
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? using ids how do you predict ???)
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test-x_raw.mean()
    mae = (actual-pred).abs().mean()
    return mae

```

```python
mae = predict_item_item(data,data.loc[1])
mae
```

```python
mae = predict_user_user(data,data.loc[1])
mae
```

## Exercise 4 (Worth 5-10 extra credit points for one or both implementions)
Combine in sequence item-item and user-user AND/OR user-user and item-item.

```python
%%capture
## YOUR SOLUTION HERE
```

```python

```
