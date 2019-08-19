#!/usr/bin/env python
# coding: utf-8

# In[75]:


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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, Normalizer, StandardScaler, RobustScaler
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


df = pd.read_pickle('df_ready1')


# In[63]:


df['target'] = 0


# In[64]:


# Setting mock federal standard for a highschool grad rate of at least 90%
df.loc[df['High school graduation raw value'] >= 0.9, 'target'] = 1  # Meeting standard

df.loc[df['High school graduation raw value'] <0.9, 'target'] = 0  # Failing to meet standard


# In[65]:


df['target'].value_counts().sort_index()


# In[66]:


(df.corr() > 0.7).sum()


# In[67]:


drop = ['Poor physical health days raw value', 'Poor mental health days raw value',
        'Adult smoking raw value', 'Children in poverty raw value',
        'Premature age-adjusted mortality raw value',
        'Frequent physical distress raw value', 'Frequent mental distress raw value']

df_3_max_corr = df.drop(drop, axis=1)


# In[68]:


(df_3_max_corr.corr() > 0.7).sum()


# In[69]:


more_to_drop = ['Physical inactivity raw value', 'Uninsured raw value', 'High school graduation raw value',
                'Severe housing problems raw value', 'Percentage of households with high housing costs',
                'Diabetes prevalence raw value', 'Uninsured adults raw value',
                'Uninsured children raw value', 'Severe housing cost burden raw value',
                '% Hispanic raw value', '% not proficient in English raw value' ]

df_1_max_corr = df_3_max_corr.drop(columns=more_to_drop, axis=1)


# In[70]:


(df_1_max_corr.corr() > 0.7).sum()


# In[71]:


df_1_max_corr.iloc[:, 5:].head()


# # PCA Analysis

# In[89]:


df_1_max_corr.head()


# # PCA scaled properly

# In[90]:


# Did not scale before eveything. Leaving the results from not scaling below.

X_pca_strict = df_1_max_corr.drop(columns='target', axis=1)
X_pca = X_pca_strict.iloc[:, 5:]
y_pca = df_1_max_corr['target']
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y_pca)

standard = StandardScaler()
X_pca_train = standard.fit_transform(X_pca_train)
X_pca_test = standard.fit_transform(X_pca_test)
X_pca_all = standard.fit_transform(X_pca)

pca = PCA(n_components=5)
df_pca_strict = pca.fit_transform(X_pca_all)
y = df_1_max_corr['target']


# In[91]:


pca.explained_variance_ratio_.cumsum()


# In[191]:


plt.figure(figsize=(15,10))
plt.plot(range(1, len(pca.explained_variance_ratio_.cumsum())+1),
               pca.explained_variance_ratio_.cumsum())

plt.scatter(range(1, len(pca.explained_variance_ratio_.cumsum())+1),
                  pca.explained_variance_ratio_.cumsum(), c='orange')

plt.title('Total Variance Explained by Varying Number of Principle Components', fontsize=20);
plt.xlabel('Number of Principle Components', fontsize=20)
plt.ylabel('Variance of Dataset', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PCA_variance_vs_components.png')


# In[127]:


dataFrame = pd.DataFrame(X_pca_all)
PC1 = dataFrame[1].values
PC2 = dataFrame[2].values


# In[125]:


y_pca_values = y_pca.values


# In[135]:


pca_df_plot = pd.DataFrame([PC1, PC2, y_pca_values]).T
pca_df_plot['target'] = pca_df_plot.iloc[:, 2].astype('int')


# In[138]:


pca_df_plot.drop(columns=2,inplace=True)


# In[139]:


pca_df_plot.columns = ['PC1', 'PC2', 'target']


# In[143]:


pca_df_plot.head()


# In[193]:


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
for target, color in zip(targets, colors):
    indicesToKeep = pca_df_plot['target'] == target
    ax.scatter(pca_df_plot.loc[indicesToKeep, 'PC1']
               , pca_df_plot.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(labels, fontsize=20, frameon=True, framealpha=1, edgecolor='k')
ax.grid()
plt.savefig('2D_graph_PC1_PC2.png')


# In[202]:


#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_pca_train, y_pca_train)
y_pred_knn = knn.predict(X_pca_test)
y_pred_knn

print('Accuracy:' + str(accuracy_score(y_pca_test, y_pred_knn)))
print('F1: ' + str(f1_score(y_pca_test, y_pred_knn)))
print(classification_report(y_pca_test, y_pred_knn))


#to find optimal k
k_range = list(range(1, 10))
k_scores = []

for k in k_range:
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_pca_train, y_pca_train)
   y_predict = knn.predict(X_pca_test)
   score = f1_score(y_pca_test, y_predict, average='weighted')
   k_scores.append( score)
print(k_scores)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(k_range, k_scores, linestyle='solid', marker='o',
        markerfacecolor='orange', markersize=10)
plt.title('F1 score by K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
plt.savefig('F1_vs_k_knn.png')
plt.show()


# In[205]:


plt.figure(figsize=(20,10))
visualizer = ROCAUC(knn)
visualizer.score(X_pca_test, y_pca_test)  # Evaluate the model on the test data
plt.savefig('ROC_curve_knn.png')
plt.title('ROC Curves for KNN Classifier')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
visualizer.poof()


# In[204]:


plt.figure(figsize = (15,10))
cm = ConfusionMatrix(knn)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(X_pca_test, y_pca_test)

# How did we do?
plt.savefig('confusion_matrix_knn.png')
cm.poof()


# In[208]:


#Support Vector Machine Classification
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_pca_train, y_pca_train)
y_pred = svclassifier.predict(X_pca_test)
print('Accuracy:' + str(accuracy_score(y_pca_test, y_pred)))
print('F1: ' + str(f1_score(y_pca_test, y_pred)))
print(classification_report(y_pca_test, y_pred))


# In[197]:


plt.figure(figsize=(20,10))
visualizer = ROCAUC(svclassifier, micro=False, macro=False, per_class=False,)
visualizer.score(X_pca_test, y_pca_test)  # Evaluate the model on the test data

visualizer.poof()
plt.savefig('ROC_svm.png')


# In[198]:


plt.figure(figsize = (15,10))
cm = ConfusionMatrix(svclassifier)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(X_pca_test, y_pca_test)

# How did we do?
cm.poof()
plt.savefig('confusion_matrix_svm.png')


# # PCA not scaled properly, but good scores

# In[72]:


# Making PC's from these data frames to try on the model
# Will call the data frame with 1 max correlation of 0.7 or above
# df_pca_strict
# The other, which allows for 3 variables to correlate greater than 0.7 will
# be called df_pca_less_strict
X_pca_strict = df_1_max_corr.drop(columns='target', axis=1)
y_pca = df_1_max_corr['target']

X_pca = X_pca_strict.iloc[:, 5:]
pca = PCA(n_components=5)
df_pca_strict = pca.fit_transform(X_pca)
y = df_1_max_corr['target']


# In[73]:


pca.explained_variance_ratio_.cumsum()


# In[78]:


plt.figure(figsize=(15,10))
plt.scatter(range(1, len(pca.explained_variance_ratio_.cumsum())+1), pca.explained_variance_ratio_.cumsum())
plt.title('Total Variance Explained by Varying Number of Principle Components', fontsize=20);
plt.xlabel('Number of Principle Components', fontsize=20)
plt.ylabel('Variance of Dataset', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


# In[18]:


X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, random_state=10, test_size = 0.2)


# In[19]:


standard = StandardScaler()
X_pca_train = standard.fit_transform(X_pca_train)
X_pca_test = standard.fit_transform(X_pca_test)


# In[20]:


#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_pca_train, y_train)
y_pred_knn = knn.predict(X_pca_test)
y_pred_knn

print('Accuracy:' + str(accuracy_score(y_test, y_pred_knn)))
print('F1: ' + str(f1_score(y_test, y_pred_knn)))
print(classification_report(y_test, y_pred_knn))


#to find optimal k
k_range = list(range(1, 10))
k_scores = []

for k in k_range:
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_pca_train, y_train)
   y_predict = knn.predict(X_pca_test)
   score = f1_score(y_test, y_predict, average='weighted')
   k_scores.append( score)
print(k_scores)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(k_range, k_scores, linestyle='solid', marker='o',
        markerfacecolor='orange', markersize=10)
plt.title('F1 score by K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
plt.show()


# In[21]:


#Support Vector Machine Classification
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_pca_train, y_train)
y_pred = svclassifier.predict(X_pca_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'The accuracy score is {accuracy_score(y_test, y_pred)}')


# In[79]:


# Again, but for df_pca_less_strict
X_pca_strict = df_3_max_corr.drop(columns='target', axis=1)
X_pca = X_pca_strict.iloc[:, 5:]
pca = PCA(n_components=5)
df_pca_strict = pca.fit_transform(X_pca)
y = df_3_max_corr['target']


# In[80]:


plt.figure(figsize=(15,10))
plt.scatter(range(1, len(pca.explained_variance_ratio_.cumsum())+1), pca.explained_variance_ratio_.cumsum())
plt.title('Total Variance Explained by Varying Number of Principle Components', fontsize=20);
plt.xlabel('Number of Principle Components', fontsize=20)
plt.ylabel('Variance of Dataset', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


# In[81]:


X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, random_state=10, test_size = 0.2)


# In[82]:


standard = StandardScaler()
X_pca_train = standard.fit_transform(X_pca_train)
X_pca_test = standard.fit_transform(X_pca_test)


# In[34]:


#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_pca_train, y_train)
y_pred_knn = knn.predict(X_pca_test)
y_pred_knn

print('Accuracy:' + str(accuracy_score(y_test, y_pred_knn)))
print('F1: ' + str(f1_score(y_test, y_pred_knn)))
print(classification_report(y_test, y_pred_knn))


#to find optimal k
k_range = list(range(1, 10))
k_scores = []

for k in k_range:
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_pca_train, y_train)
   y_predict = knn.predict(X_pca_test)
   score = f1_score(y_test, y_predict, average='weighted')
   k_scores.append( score)
print(k_scores)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(k_range, k_scores, linestyle='solid', marker='o',
        markerfacecolor='orange', markersize=10)
plt.title('F1 score by K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
plt.show()


# In[35]:


#Support Vector Machine Classification
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_pca_train, y_train)
y_pred = svclassifier.predict(X_pca_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'The accuracy score is {accuracy_score(y_test, y_pred)}')


# # Raw Data Analysis

# In[22]:


X = df.iloc[:, 6:]


# In[23]:


# Making features and target
X.drop(columns=['High school graduation raw value', 'target'], inplace=True)
X = X.values
y = df['target'].values


# In[24]:


# train, test, split variables
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size = 0.2)


# In[25]:


standard = StandardScaler()
X_train_scaled = standard.fit_transform(X_train)
X_test_scaled = standard.fit_transform(X_test)
X_scaled = standard.fit_transform(X)


# In[26]:


#Support Vector Machine Classification
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'The accuracy score is {accuracy_score(y_test, y_pred)}')


# In[27]:


#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn

print('Accuracy:' + str(accuracy_score(y_test, y_pred_knn)))
print('F1: ' + str(f1_score(y_test, y_pred_knn)))
print(classification_report(y_test, y_pred_knn))


#to find optimal k
k_range = list(range(1, 10))
k_scores = []

for k in k_range:
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_train, y_train)
   y_predict = knn.predict(X_test)
   score = f1_score(y_test, y_predict, average='weighted')
   k_scores.append( score)
print(k_scores)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(k_range, k_scores, linestyle='solid', marker='o',
        markerfacecolor='orange', markersize=10)
plt.title('F1 score by K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
plt.show()


# In[ ]:


# XG BOOST
import xgboost as xgb
xgc_scores = {}
xgc_feature_sets = []
k=0
for i in range(0, 100, 10):
    for j in range(0, 100, 10):
        xgc = xgb.XGBClassifier(reg_alpha=i, reg_lambda=j)
        xgc.fit(X_train_scaled, y_train)


        xgc_features = pd.merge(pd.DataFrame(xgc.feature_importances_,
                     index=X.iloc[:, 2:].columns,
                     columns=["importance"]
                    ).sort_values(by="importance", ascending=False), var_description, how="left",
                                right_index=True, left_index=True)
        xgc_scores[k] = [i,
                         j,
                         xgc.score(X_train_scaled, y_train),
                         xgc.score(X_test_scaled, y_test),
                         np.sum(xgc_features.importance>0)]
        xgc_feature_sets.append(xgc_features)
        k += 1


# In[ ]:
