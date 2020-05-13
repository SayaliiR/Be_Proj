#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from csv import writer
from csv import DictWriter
import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"video_Data23.csv", index_col=None, sep=',')


# In[3]:


df


# In[4]:


data=df.loc[:,['gConfId','viewer','publishedDuration','timeSpent','channels','topics']]
data


# In[5]:


data.dtypes


# In[6]:


data['ratioTP'] = data['timeSpent']/data['publishedDuration']
data


# In[7]:


import random
import operator
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler


# In[8]:


data['videoID'] = data['gConfId'].astype("category").cat.codes
data['viewerID'] = data['viewer'].astype("category").cat.codes


# In[9]:


data


# In[11]:


data = data.drop(['gConfId', 'viewer'], axis=1)


# In[12]:


data = data.loc[data.ratioTP != 0.0]
data


# In[13]:


viewers = list(np.sort(data.viewerID))
videos = list(np.sort(data.videoID))
timeSpent= list(data.ratioTP)
len(viewers)


# In[14]:


rows = data.viewerID.astype(int)
cols = data.videoID.astype(int)
type(rows)
type(cols)
rows


# In[15]:


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


# In[16]:


# The implicit library expects data as a item-user matrix so we
# create two matricies, one for fitting the model (item-user) 
# and one for recommendations (user-item)
sparse_item_user = sparse.csr_matrix((data['timeSpent'].astype(float), (data['videoID'], data['viewerID'])))
sparse_user_item = sparse.csr_matrix((data['timeSpent'].astype(float), (data['viewerID'], data['videoID'])))


# In[17]:


product_train, product_test, product_users_altered = make_train(sparse_item_user, pct_test = 0.2)#Training data set for fitting the model


# In[18]:


product_train1, product_test, product_users_altered = make_train(sparse_user_item, pct_test = 0.2)#Training data set for recommendation


# In[19]:


import sys
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
from sklearn.preprocessing import MinMaxScaler
import implicit 


# In[20]:




# Initialize the als model and fit it using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
data_conf = (product_train* alpha_val).astype('double')

#Fit the model
model.fit(data_conf)
user_vecs = model.user_factors
item_vecs = model.item_factors
    
    

def get_reccomendation(viewer_id,product_train1):
    recommended = model.recommend(int(viewer_id),product_train1)
    videos = []
    video_scores = []

    for item in recommended:
        idx, score = item
        videos.append(data.gConfId.loc[data.videoID == str(idx)].iloc[0])
        video_scores.append(score)
    
    recommendations =pd.DataFrame({'video_recommended': videos, 'score': video_scores})
    return recommendations 







   



    
    
















