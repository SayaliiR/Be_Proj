#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from csv import writer
from csv import DictWriter
import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\Lenovo\Desktop\BE PROJECT/video_Data23.csv",sep=',')


# In[3]:


df


# In[4]:


data=df.loc[:,['gConfId','viewer','publishedDuration','timeSpent','channels','topics']]
data


# In[5]:


data['ratioTP'] = data['timeSpent']/data['publishedDuration']
data


# In[6]:


import random
import operator
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler


# In[7]:


data['videoID'] = data['gConfId'].astype("category").cat.codes
data['viewerID'] = data['viewer'].astype("category").cat.codes


# In[8]:


data


# In[9]:


data = data.loc[data.ratioTP != 0.0]
data


# In[10]:


viewers = list(np.sort(data.viewerID))
videos = list(np.sort(data.videoID))
timeSpent= list(data.ratioTP)
len(viewers)


# In[11]:


rows = data.viewerID.astype(int)
cols = data.videoID.astype(int)
type(rows)
type(cols)
rows


# In[12]:


def make_train(ratings, pct_test = 0.2):
   
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  


# In[13]:


# The implicit library expects data as a item-user matrix so we
# create two matricies, one for fitting the model (item-user) 
# and one for recommendations (user-item)
sparse_item_user = sparse.csr_matrix((data['timeSpent'].astype(float), (data['videoID'], data['viewerID'])))
sparse_user_item = sparse.csr_matrix((data['timeSpent'].astype(float), (data['viewerID'], data['videoID'])))


# In[14]:


product_train, product_test, product_users_altered = make_train(sparse_item_user, pct_test = 0.2)#Training data set for fitting the model


# In[15]:


product_train1, product_test, product_users_altered = make_train(sparse_user_item, pct_test = 0.2)#Training data set for recommendation


# In[16]:


import implicit


# In[17]:


model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)


# In[18]:



# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
data_conf = (product_train* alpha_val).astype('double')

#Fit the model
model.fit(data_conf)
user_vecs = model.user_factors
item_vecs = model.item_factors


# In[19]:


from sklearn import metrics


# In[20]:


def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr) 


# In[21]:


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
   # Return the mean AUC rounded to three decimal places for both test and popularity benchmark


# In[22]:


calc_mean_auc(product_train1, product_users_altered, 
              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], product_test)


# In[ ]:




