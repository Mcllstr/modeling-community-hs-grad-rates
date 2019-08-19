#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read in data
df = pd.read_csv('analytic_data2019.csv', low_memory=False)


# In[3]:


df.head(10) # Has a lot of NaN and repeated values


# In[4]:


df.head(-10) # Last several columns are looking pretty absent


# In[5]:


list(df.columns)


# In[6]:


# Getting rid of everything except for raw data columns or other things that seem important
headers_ = [x for x in df.columns if 'raw value' in x
           or 'FIPS' in x
           or 'State Abbreviation' in x
           or 'Name' in x
           or 'County Ranked' in x
           or 'Percentage of households' in x
           ]

df = df[headers_]


# In[7]:


df.info() # 114 features


# In[8]:


list(df.columns)


# In[9]:


report = pandas_profiling.ProfileReport(df)


# In[10]:


report


# In[11]:


df.isna().sum()


# In[12]:


msno.matrix(df)


# In[13]:


# Handling missing data.


# In[14]:


# Dropping anything with about half or more of its rows missing
cols_to_drop = list(df.isna().sum()[df.isna().sum() > 1600].index)
df.drop(columns=cols_to_drop, inplace=True)
msno.matrix(df)


# In[15]:


total_columns_missing_data = len(df.isna().sum())
total_rows = 3195
total_columns = len(list(df.columns))
print(total_columns_missing_data, '\n', total_rows, '\n', total_columns)
# All columns are missing data.
# Since the target variable is High school graduation rates
# We can see how much of that is missing.
df['High school graduation raw value'].isna().sum()/len(df)
# 3.1 % is missing.


# In[16]:


# If we drop all columns missing more than 3.3% of data how much will we drop?
df.isna().sum()[df.isna().sum() > 0.033*3195]


# In[17]:


# Ok anything missing more than 10% of its data points is out
cols_to_drop = df.isna().sum()[df.isna().sum() > 320].index
df.drop(columns=cols_to_drop, axis=1, inplace=True)


# In[18]:


df.isna().sum()[df.isna().sum() > 0.033*3195]

# What to do about the rest? Going to drop


# In[19]:


df10 = df.copy() # making a 10% dataframe
report = pandas_profiling.ProfileReport(df10)
report


# In[34]:


pd.to_pickle(df10, 'still_missing_data_df')


# In[25]:


# Making a stricter dataframe that cuts off where the target variable does
# i.e. anything missing more than 3.2% of data is out
cols_to_drop = df.isna().sum()[df.isna().sum() > df['High school graduation raw value'].isna().sum()].index
strict_df = df.drop(columns=cols_to_drop)
strict_df.isna().sum().max() # dropping missing values from this

strict_df.dropna(inplace=True)
strict_df.isna().sum()
pd.to_pickle(strict_df, 'df0')


# In[27]:


col_names = list(strict_df.columns)
number_of_zeros = (strict_df[col_names]=='0').sum()
number_of_zeros.max()
# there are a lot of zeros going on


# In[31]:


type(strict_df['High school graduation raw value'][0]) # These things are strings.


# In[32]:


strict_df.describe()


# In[35]:


report = pandas_profiling.ProfileReport(strict_df)


# In[36]:


report


# In[53]:


strict_df.head() # Have to drop the first row
test = strict_df.drop(0, axis=0)
test


# In[54]:


def data_frame_float_converter(x):
    try:
        return x.astype('float')
    except:
        return x


# In[55]:


data_type = test.iloc[:, 7:].apply(data_frame_float_converter, axis=0)


# In[56]:


type(data_type['Poor or fair health raw value'].iloc[1])


# In[58]:


df_floats = pd.concat([test.iloc[:, :7], data_type], axis=1)
df_floats


# In[ ]:


# column_names = list(county_df.columns)


# In[ ]:


# df_floats = pd.DataFrame()
# for col in col_names:
#     try:
#         df_floats[col] = county_df[col].astype('float')
#     except:
#         df_floats[col] = county_df[col]



# In[63]:


df_floats.info()


# In[65]:


pd.to_pickle(df_floats, 'df_ready1')


# In[66]:


# To do: explore imputation with df10; saved as 'still_missing_data_df'
#        Think more on how you can improve this data to use in models (look at each
#        individual column and figure it out from there?)


# In[ ]:





# In[117]:


plt.figure(figsize=(30,10))
df_floats.boxplot()


# In[95]:


df_floats.head()


# In[108]:


df_floats['Adult obesity raw value'].max()


# In[111]:


columns_list = df_floats.columns
for number, col in enumerate(columns_list):
    try:
        print(col,'----->', df_floats[col].max())
    except:
        continue


# In[ ]:
