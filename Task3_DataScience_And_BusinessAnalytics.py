#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# ## DATA SCIENCE AND BUSINESS ANALYTICS TASK : 3

# ## Prediction using Decision Tree Algorithm: Create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# ## Language: Python
# ## IDE: Jupyter Notebook
# ## Libraries/Datasets used: Scikit Learn, Pandas, Pydotplus, Iris Dataset

# ### By SUPARNA SARKAR

# **Importing libraries | Loading Iris datasets | Forming the iris dataframe into notebook**

# In[1]:


# Importing libraries into notebook
import pandas as pd
import sklearn.datasets as datasets
from sklearn import tree


# In[2]:


iris = datasets.load_iris()
iris_df = pd.read_excel(r'C:\Users\Suparna\Documents\SuparnaDataset\Iris.xlsx')
print("Iris data loaded successfully" )  


# In[3]:


# Forming the iris dataframe
df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(df.head(15))

y=iris.target
print(y)


# **Defining Decision Tree Algorithm**

# In[4]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(df,y)

print("Decision Tree Classifier Created!!")


# **Visualizing the Decision Tree Created**

# In[5]:


# installing the required libraries
get_ipython().system('pip install pydotplus')
get_ipython().system('pip install graphviz')


# In[6]:


pip install six


# In[7]:


import six
import sys
sys.modules['sklearn.externals.six'] = six


# In[8]:


# importing necessary libraries for Tree Visualization
from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
print("Import Successful")


# In[9]:


pip install graphviz


# In[10]:


# Visualizing the graph
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names = iris.feature_names, filled = True, rounded = True, special_characters = True, node_ids = True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# ---End of Code---
