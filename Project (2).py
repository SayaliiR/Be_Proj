#!/usr/bin/env python
# coding: utf-8

# In[1]:



import csv
from csv import writer
from csv import DictWriter
import pandas as pd


# In[2]:


df=pd.read_csv(r"video_Data23.csv")


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
import pandas as pd
import numpy as np


import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler


# In[8]:


data['videoID'] = data['gConfId'].astype("category").cat.codes
data['viewerID'] = data['viewer'].astype("category").cat.codes


# In[9]:


data


# In[10]:


item_lookup = data[['videoID', 'gConfId']].drop_duplicates()
item_lookup['videoID'] = item_lookup.videoID.astype(str)
item_lookup.loc[0,:]


# In[11]:


data = data.drop(['gConfId', 'viewer'], axis=1)


# In[12]:


data.head(40)


# In[13]:


data = data.loc[data.ratioTP != 0.0]


# In[14]:


viewers = list(np.sort(data.viewerID))
videos = list(np.sort(data.videoID))
timeSpent= list(data.ratioTP)
len(viewers)


# In[15]:


rows = data.viewerID.astype(int)
cols = data.videoID.astype(int)
type(rows)
type(cols)
rows


# In[16]:


sparse.csr_matrix.__hash__ = object.__hash__
data_sparse = sparse.csr_matrix((timeSpent, (rows, cols)),shape=(len(viewers),len(videos)))


# In[17]:


data_sparse.toarray()


# In[18]:


def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):
    confidence = sparse_data * alpha_val
    # Calculate the foncidence for each value in our data
        
    
    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape
    
    # We create the user vectors X of size users-by-features, the item vectors
    # Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size = (user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size = (item_size, features)))
    
    #Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(item_size)
    I = sparse.eye(features)
    lI = lambda_val * I
    for i in range(iterations):
        print ('iteration %d of %d' % (i+1, iterations))
        
        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in range(user_size):

            # Get the user row.
            u_row = confidence[u,:].toarray() 

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

    
        for i in range(item_size):

            # Get the item column and transpose it.
            i_row = confidence[:,i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            # Calculate Ci and Ci - I
            CiI = sparse.diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y


# In[19]:


user_vecs, item_vecs = implicit_als(data_sparse, iterations=1, features=20, alpha_val=40)
    


# In[20]:


#------------------------------
# FIND SIMILAR ITEMS
#------------------------------

# Let's find similar videos  
item_id = 7

# Get the item row 
item_vec = item_vecs[item_id].T

# Calculate the similarity score and select the top 5 most similar.
scores = item_vecs.dot(item_vec).toarray().reshape(1,-1)[0]
top_3 = np.argsort(scores)[::-1][:3]

videos = []
video_scores = []

# Get and print the actual video
for idx in top_3:
    #item_lookup.gConfId.loc[item_lookup.videoID == str(idx)]
    videos.append(item_lookup.gConfId.loc[item_lookup.videoID == str(idx)])
    video_scores.append(scores[idx])
        
    
a = {'video': videos, 'score': video_scores}
similar = pd.DataFrame.from_dict(a, orient='index')
similar.transpose()
#similar = pd.DataFrame({'video': videos, 'score': video_scores})

print(similar)


# In[21]:


viewer_id = 5;

consumed_idx = data_sparse[viewer_id,:].nonzero()[1].astype(str)
consumed_items = item_lookup.loc[item_lookup.videoID.isin(consumed_idx)]
print(consumed_items)


# In[22]:


def recommend(viewer_id, data_sparse, user_vecs, item_vecs, item_lookup, num_items=10):
     # Get all interactions by the user
    user_interactions = data_sparse[viewer_id,:].toarray()

    # We don't want to recommend items the user has consumed. So let's
    # set them all to 0 and the unknowns to 1.
    user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
    user_interactions[user_interactions > 1] = 0

    # This is where we calculate the recommendation by taking the 
    # dot-product of the user vectors with the item vectors.
    rec_vector = user_vecs[viewer_id,:].dot(item_vecs.T).toarray()

    # Let's scale our scores between 0 and 1 to make it all easier to interpret.
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions*rec_vector_scaled
   
    # Get all the artist indices in order of recommendations (descending) and
    # select only the top "num_items" items. 
    item_idx = np.argsort(recommend_vector)[::-1][:num_items]
    
    
    
    videos = []
    scores = []

    # Loop through our recommended artist indicies and look up the actial artist name
    for idx in item_idx:
        videos.append(item_lookup.gConfId.loc[item_lookup.videoID == str(idx)].iloc[0])
        scores.append(recommend_vector[idx])

    # Create a new dataframe with recommended artist names and scores
    recommendations = pd.DataFrame({'video_recommended': videos, 'score': scores})
    
    return recommendations

# Let's generate and print our recommendations
recommendations = recommend(viewer_id, data_sparse, user_vecs, item_vecs, item_lookup)

print(recommendations)

     


# In[ ]:


import os, os.path
import cherrypy
import json


class Welcomepage(object):
    @cherrypy.expose
    def index(self):
        return '''<html>
<head>
<style>
body {
  background-color:pink;
}
</style>
</head>
<body>
<h1>
<p style="color:red">MOVIE RECCOMENDATION</p>
</h1>
            <form action="Recsys" method="GET">
            Enter viewer_id:
            <input type="text" name="viewer_id" /><br>
            Enter num_items:
            <input type="text" name="num_items" /><br>
            <input type="submit" />
            </form>
            </htm>'''
        
        #return json.dumps(recommend(viewer_id, data_sparse, user_vecs, item_vecs, item_lookup, num_items=3).to_json())
        

    @cherrypy.expose
    def Recsys(self, **params):
        return json.dumps(recommend(viewer_id, data_sparse, user_vecs, item_vecs, item_lookup, num_items=10).to_json())
        #rec = recommend(viewer_id, data_sparse, user_vecs, item_vecs, item_lookup)
       # return json.dumps(rec)
        
        

        #if viewer_id is None:
         #No name was specified
           #return 'Please enter name <a href="./">here</a>.'
       # else:
           # return 'enter user name first <a href="./">here</a>.'
        
        
        
   

       


if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './public'
        }
    }
    cherrypy.quickstart(Welcomepage(), '/', conf)


# In[ ]:





# In[ ]:




