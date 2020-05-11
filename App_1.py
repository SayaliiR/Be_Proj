#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Train_1 import * #Importing the Train_1.py


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
            <form name="search" action="Recsys" method="GET">
            Enter viewer_id:
            <input type="text" name="viewer_id">
            <button type="submit" class="button">Submit</button>
            </form>
            </html>'''

    @cherrypy.expose
    def Recsys(self,viewer_id, **params):
        return get_reccomendation(int(viewer_id),product_train1).to_json()
       

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




