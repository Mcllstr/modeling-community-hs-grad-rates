#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, Normalizer, StandardScaler, RobustScaler
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


# Read in data frame from EDA
df = pd.read_pickle('df_ready1')


# In[26]:


df.info()


# In[27]:


df.head()


# In[28]:


# View all column topics
list(df.columns)


# In[29]:


df['target'] = 0 # create a target column


# In[30]:


# Looking at the target variable distribution
sns.distplot(df['High school graduation raw value'])


# In[31]:


plt.hist(df['High school graduation raw value'])


# In[32]:


# Setting mock federal standard for a highschool grad rate of at least 90%
df.loc[df['High school graduation raw value'] >= 0.9, 'target'] = 1  # Meeting standard

df.loc[df['High school graduation raw value'] <0.9, 'target'] = 0  # Failing to meet standard


# In[33]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr())


# In[ ]:


0.21697799347796504, % rural
 0.2719691482359243, Home ownership
 0.31094167052019217, % Non-Hispanic white raw value
 0.7300252065326511

 -0.32083586573184425, Children in single-parent households raw value
 -0.31468499753911716, Severe housing problems raw value
 -0.30586254195786583, Percentage of households with high housing costs
 -0.3017989466551982,
 -0.24732405190486392,


# In[90]:


df.corr()[df.corr()['target'] == -0.30586254195786583]


# In[34]:


df.corr()


# In[35]:


df['target'].value_counts().sort_index()


# In[92]:


len(df.columns)


# In[36]:


# Making predictors and predictions
X = df.iloc[:, 6:]


# In[37]:


# Making features and target
X.drop(columns=['High school graduation raw value', 'target'], inplace=True)
X = X.values
y = df['target'].values


# In[38]:


# train, test, split variables
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size = 0.2)


# In[39]:


# Scaling data
standard = StandardScaler()
X_train_scaled = standard.fit_transform(X_train)
X_test_scaled = standard.fit_transform(X_test)
X_scaled = standard.fit_transform(X)


# In[40]:


# PCA components for training data
pca = PCA(0.99)
X_pca_train = pca.fit_transform(X_train_scaled)

PC1_train, PC2_train = X_pca_train[:, 0], X_pca_train[:, 1]


# In[41]:


# PCA components for testing data
pca = PCA(0.99)
X_pca_test = pca.fit_transform(X_test_scaled)

PC1_test, PC2_test = X_pca_test[:, 0], X_pca_test[:, 1]


# In[42]:


# PCA components for all data
pca = PCA(0.99)
X_pca = pca.fit_transform(X_scaled)

# Grabbing the first two PCs
PC1, PC2 = X_pca[:, 0], X_pca[:, 1]


# In[43]:


pca_length = len(pca.explained_variance_ratio_.cumsum())


# In[93]:


# Plot PCA CDF
plt.figure(figsize=(15,10))
plt.scatter(range(1,pca_length+1), pca.explained_variance_ratio_.cumsum())
plt.title('Total Variance Explained by Varying Number of Principle Components', fontsize=20);
plt.xlabel('Number of Principle Components', fontsize=20)
plt.ylabel('Variance of Dataset', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PCA1_raw_data.png')


# In[45]:


pca.explained_variance_ratio_.cumsum()[:42] # the first 42 PC's capture 99% of data


# In[46]:


# Creating a PCA dataframe with all data to observe how it splits the groups
# This is needed for the following plot
PCA_df = pd.DataFrame(X_pca)
y = pd.Series(y)


# In[47]:


PCA_df['target'] = y
col_labels = ['PC' + str(x) for x in range(1,43)]
col_labels.append('target')
PCA_df.columns = col_labels
PCA_df['target'] = PCA_df['target'].astype('int')


# In[48]:


PCA_df.head()


# In[49]:


target = PCA_df['target']


# In[94]:


fig = plt.figure(figsize = (15,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
targets = [0, 1]
labels = ['NOT meeting standard', 'Meeting Standard']
colors = ['orange', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = PCA_df['target'] == target
    ax.scatter(PCA_df.loc[indicesToKeep, 'PC1']
               , PCA_df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(labels, fontsize=20, frameon=True, framealpha=1, edgecolor='k')
ax.grid()
plt.savefig('PCA2_first_two_comps_plot.png')


# # Random Forest
#
#

# In[55]:


# First running a random forest on our actual data
clf = RandomForestClassifier()
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)


# In[56]:


accuracy_score(y_test, y_pred)


# In[57]:


confusion_matrix(y_test, y_pred)


# In[58]:


# Instantiate the visualizer with the classification model
plt.figure(figsize=(20,10))
visualizer = ROCAUC(clf)
visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.poof()


# In[63]:


print(classification_report(y_test, y_pred))


# In[64]:


X_pca_train_random_forest = X_pca_train[:, :-1]


# In[65]:


# Trying again, but this time with PC's
# X_pca_train has 42 columns while X_pca_test has 41.
# This throws an error.
# Dropping one of the PC's from X_pca_train to geet around this

clf = RandomForestClassifier()
clf.fit(X_pca_train_random_forest, y_train)
y_pred = clf.predict(X_pca_test)


# In[66]:


plt.figure(figsize=(20,10))
visualizer = ROCAUC(clf)
visualizer.score(X_pca_test, y_test)  # Evaluate the model on the test data
visualizer.poof()


# In[68]:


plt.figure(figsize=(15,10))
cm = ConfusionMatrix(clf)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(X_pca_test, y_test)

# How did we do?
cm.poof()


# In[69]:


accuracy_score(y_test, y_pred)


# In[70]:


print(classification_report(y_test, y_pred))


# In[49]:


steps = [('forest', RandomForestClassifier())]
pipe = Pipeline(steps=steps)

grid = {'forest__n_estimators': list(np.arange(1, 300, 20)),
        'forest__criterion': ['gini', 'entropy'],
        'forest__max_depth': list(np.arange(1, 7, 1)),
        'forest__max_leaf_nodes': list(np.arange(2, 100, 10)),
        'forest__random_state' : [0]}

gs = GridSearchCV(pipe, grid, cv=5, scoring='accuracy')


gs.fit(X_train_scaled, y_train)
gs.score(X_test_scaled, y_test)
gs.best_params_


# In[71]:


# First running a random forest on our actual data
clf = RandomForestClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=52, n_estimators=221)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)


# In[72]:


accuracy_score(y_test, y_pred)


# In[73]:


print(classification_report(y_test, y_pred))


# In[74]:


# Instantiate the visualizer with the classification model
plt.figure(figsize=(20,10))
visualizer = ROCAUC(clf)
visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.poof()


# In[75]:


plt.figure(figsize=(15,10))
cm = ConfusionMatrix(clf)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(X_test, y_test)

# How did we do?
cm.poof()


# In[ ]:
