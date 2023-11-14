#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install ucimlrepo')

get_ipython().system('pip3 install -U ucimlrepo')

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

#importing the dataset of adult and its info



# In[2]:


get_ipython().run_line_magic('lsmagic', '')
#showing what commands are possible within our program


# In[3]:


X.head()

#checking the head and seeing the differences between them in terms of age m gender etc


# In[4]:


X.info()
#showing the info of adult and seeing how many are have data or not


# In[5]:


X.describe()


# In[6]:


X.shape
#shape of X


# In[7]:


X.hist(figsize=(24, 16))
plt.show()
#showing the data of X in the form of a histograph for each catagory


# In[8]:


pd.Series((X.values== "?").sum(axis=0), index= (X.columns))
#checking to see how many missing values we have 


# In[9]:


(X.values =='?').sum()
X = X.replace('?', np.nan)
X.isna().sum()
#replacing our missing values with nan


# In[10]:


X.info()
#checking to see our updated info


# In[11]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#creating a new pipleline that will do the following: fill missing values , scale numerical colmns , fill missing categorical values and 
#encode the categorical columns


# In[12]:


#create the cat and num colmns
#get a list of columns from x that are of numerical data types
#get a list of columns from x dataFrame that are not of numerical data types.

num_cols = X.select_dtypes(include='number').columns.to_list()
cat_cols = X.select_dtypes(exclude='number').columns.to_list()

# num_cols.remove("hincome")

num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())
preprocessing = ColumnTransformer([('num', num_pipeline, num_cols),
                                   ('cat', cat_pipeline, cat_cols)],
                                    remainder='passthrough'
                                 )


# In[13]:


OneHotEncoder(sparse_output=False)
preprocessing
#printing out our preprocessing pipeline


# In[14]:


# Apply the preprocessing pipeline on the dataset

X_prepared = preprocessing.fit_transform(X)


X_prepared.shape


# In[15]:


#checking the data withing value_counts the target in this case is income
y.value_counts()


# In[16]:


#removing the dot
adult.data.targets.loc[:, 'income'] = adult.data.targets['income'].str.replace('.', '')


# In[17]:


print(y.value_counts())
#reprinting y.values to see the result 


# In[18]:


from sklearn.model_selection import train_test_split

X = adult.data.features
y = pd.DataFrame(adult.data.targets['income'], columns=['income'])

# Split the data into 80% training set and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[19]:


print(y.head())


# In[22]:


from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split

model_svm = SVC(kernel='poly', C=0.1, gamma=1)
model_svm.fit(X_train[:10000], y_train[:10000].values.ravel())


# In[ ]:


y_pred = model_svm.predict(X_test_encoded)

# Print the classification report
print(classification_report(y_test, y_pred))


# In[ ]:


conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=model_svm.classes_)
disp.plot()


# In[ ]:


param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'gamma': [0.1, 1, 10],
              'C': [0.1, 1, 10]}

# Create the GridSearchCV object
grid_search = GridSearchCV(svm, param_grid, cv=3)

# Fit the model to the data
grid_search.fit(X_train_encoded, y_train.values.ravel())

# Print the best parameters
print("Best parameters found :] : ", grid_search.best_params_)


# In[ ]:


#training the model under a 60% training , 20% validation and 20% testing
X_train,X_validation_test,y_train,y_validation_test = train_test_split(X_prepared, y, test_size=0.4, random_state=42)
X_validation,X_test,y_validation,y_test = train_test_split(X_validation_test,y_validation_test,test_size=0.5, random_state=42)
print(X_train.shape, y_train.shape,X_validation.shape, y_validation.shape, X_test.shape, y_test.shape)


# In[ ]:


from sklearn.model_selection import GridSearchCV
svm_parameter = {'kernel' : ['rbf'], 'C': [0.01, 0.1,1,10], 'gama': [0.01,1,10]}
svm =SVC()
svm_gs =GridSearchCV(estimator = svm,
                     param_grid = svm_parameters)
svm_gs.fit(X_train.iloc[:10000],y_train.iloc[:10000].values.ravel())

svm_winner = svm_gs.best_estimator_
svm_winner.score(X_validation, y_validation)


# In[ ]:





# In[ ]:




