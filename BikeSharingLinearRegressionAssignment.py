#!/usr/bin/env python
# coding: utf-8

# ## Boom Bikes Rental Case Study
# 
# ### Problem Statement
# 
# A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. Company wants to understand the factors affecting the demand for these shared bikes in the American market.
# 
# ### Key Objectives
# 
# Company wants to understand the factors affecting the demand for these shared bikes in the American market
# - Which variables are significant in predicting the demand for shared bikes.
# - How well those variables describe the bike demands.
# 
# 
# ### Business Goals
# 
# Build a model the demand for shared bikes with the available independent variables so that management wants
# - To understand how exactly the demands vary with different features. 
# - Use the model to manipulate the business strategy to meet the demand levels and meet the customer's expectations. 

# In[1]:


# Importing required libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ## Exploratory Data Analysis

# In[2]:


# reading the data file
bikes_demand = pd.read_csv('C:/Ashwani/Upgrad/Machine Learning/Linear Regression Assignment/day.csv')


# In[3]:


bikes_demand.head()


# In[4]:


bikes_demand.shape


# In[5]:


bikes_demand.info()


# No Null values found

# In[6]:


bikes_demand.describe()


# ## Visualising the Data
# Working on one of the most important step - understanding the data. We'll visualise our data using matplotlib and seaborn.
#  
# Here's where you'll also identify if some predictors directly have a strong association with the outcome variable. If there is some obvious multicollinearity going on, this is the first place to catch it 
# 

# ### Visualising Numeric Variables
# Pairplot of all the numeric variables

# In[7]:


sns.pairplot(bikes_demand)
plt.show()


# In[8]:


# check correlation coefficients to see which variables are highly correlated
plt.figure(figsize= (20,12))
sns.heatmap (bikes_demand.corr(), annot = True, cmap="YlGnBu")
plt.show()


# # Visualising Categorical Variables
#  Let's map variables and make a boxplot for some of these variables.

# In[9]:


# mapping of Categorical variables 
bikes_demand[["season"]]=bikes_demand[["season"]].apply(lambda x:x.map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"}))
bikes_demand[['mnth']]=bikes_demand[['mnth']].apply(lambda x:x.map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5: "May", 6:"Jun",7:"Jul",8:"Aug",9:"Sept",10:"Oct",11: "Nov",12:"Dec"}))
bikes_demand[['weathersit']] = bikes_demand[['weathersit']].apply(lambda x:x.map({1:'ClearClouds',2:"MistCloudy",3:"LightSnow",4:"HeavyRain"}))
bikes_demand[['weekday']] = bikes_demand[['weekday']].apply(lambda x:x.map({0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat"}))
bikes_demand[['holiday']] = bikes_demand[['holiday']].apply(lambda x:x.map({0:"No",1:"Yes"}))
bikes_demand[['workingday']] = bikes_demand[['workingday']].apply(lambda x:x.map({0:"No",1:"Yes"}))
bikes_demand[['yr']] = bikes_demand[['yr']].apply(lambda x:x.map({0:"2018",1:"2019"}))


# In[10]:


bikes_demand.head()


# In[11]:


plt.figure(figsize=(16, 16))
plt.subplot(2,2,1)
sns.boxplot(x = 'holiday', y = 'cnt', data = bikes_demand)
plt.subplot(2,2,2)
sns.boxplot(x = 'workingday', y = 'cnt', data = bikes_demand)
plt.subplot(2,2,3)
sns.boxplot(x = 'yr', y = 'cnt', data = bikes_demand)
plt.subplot(2,2,4)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bikes_demand)
plt.show()


# In[12]:


#We can also visualise some of these categorical features parallely by using the hue argument. 

plt.figure(figsize = (15, 6))
sns.boxplot(x = 'holiday', y = 'cnt', hue = 'season', data = bikes_demand)
plt.show()


# In[13]:


plt.figure(figsize = (24, 8))
sns.boxplot(x = 'holiday', y = 'cnt', hue = 'mnth', data = bikes_demand)
plt.show()


# ## Observations based on initial Visualization
# - Demand is high during Working days
# - Demand is high during Fall and Summer seasons ( more specifically in the months of June, Aug and Sept)
# - Demand is high when weather is Clear, less or partly cloudy
# - Demand is high in 2019 compared to 2018
# - Demand increases with temperature
# - Demand decreases with humidity and windspeed
# 
# 
# - *atemp* and *temp* are highly correlated
# - *cnt* and *registered* are highly correlated
# - *season* and *mnth* are highly correlated
# 

# ## Dropping variables based on EDA and Visualization
# 
# - *instant* is just an record index, so can be dropped.
# - *cnt* (target variable) is total casual and registered, hence those two can be dropped.
# - *dteday* can be dropped, as other date variables like month, weekday, year etc. are present.
# - *atemp* and *temp* are highly correlated, so dropping atemp
# - *season* and *mnth* are highly correlated and from box plots it is also clear that months within season do not show huge variation, so dropping mnth
# 

# In[14]:


bikes_demand = bikes_demand.drop(['instant', 'dteday','casual', 'registered', 'atemp', 'mnth'], axis = 1)


# In[15]:


bikes_demand.head()


# # Visualizing Numerical Values again

# In[16]:


sns.pairplot(bikes_demand)
plt.show()


# In[17]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize= (16,8))
sns.heatmap (bikes_demand.corr(), annot = True, cmap="YlGnBu")
plt.show()


# ## Similar observations found
# - Demand goes high with rise in temperature
# - Demand is inversely affected by humidity
# - Demand is inversely affected by windspeed

# ## Step 3: Data Preparation
# You can see that your dataset has many columns categorical values. But in order to fit a regression line, we would need numerical values again and not string. 

# ### Dummy Variables
# We need to convert categorical values into integer as well. For this, we will use dummy variables.

# In[18]:


## Dummy columns creation

df_season = pd.get_dummies(bikes_demand['season'], drop_first=True)
df_weather = pd.get_dummies(bikes_demand['weathersit'], drop_first=True)
df_weekday=pd.get_dummies(bikes_demand['weekday'], drop_first=True)
df_year = pd.get_dummies(bikes_demand['yr'], drop_first=True, prefix='Year')
df_holiday= pd.get_dummies(bikes_demand['holiday'], drop_first=True, prefix='Holiday')
df_workingday = pd.get_dummies(bikes_demand['workingday'], drop_first=True, prefix='Workingday')


# In[19]:


print(df_season)
print(df_workingday)


# In[20]:


bikes_demand = pd.concat([bikes_demand, df_season], axis=1)
bikes_demand = pd.concat([bikes_demand, df_weather], axis=1)
bikes_demand = pd.concat([bikes_demand, df_weekday], axis=1)
bikes_demand = pd.concat([bikes_demand, df_year], axis=1)
bikes_demand = pd.concat([bikes_demand, df_holiday], axis=1)
bikes_demand = pd.concat([bikes_demand, df_workingday], axis=1)


# In[21]:


bikes_demand.head()


# In[22]:


# Droping categorical columns post dummy column creation
bikes_demand.drop(['season','yr', 'weathersit', 'workingday', 'holiday', 'weekday'], inplace=True, axis=1)


# In[23]:


bikes_demand.head()


# In[24]:


bikes_demand.info()


# ## Splitting the Data into Training and Testing Sets
# As you know, the first basic step for regression is performing a train-test split.

# In[25]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(bikes_demand, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[26]:


print(df_train.shape)
print(df_test.shape)


# ### Rescaling the Features
#  it is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients. We will use MinMax scaling.

# In[27]:


from sklearn.preprocessing import MinMaxScaler


# In[28]:


scaler = MinMaxScaler()


# In[29]:


df_train.head()


# In[30]:


# Apply scaler() to all the columns except the '0-1' and 'dummy' variables
num_vars = ['temp', 'hum', 'windspeed', 'cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[31]:


df_train.head()


# In[32]:


df_train.describe()


# In[33]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize= (20,12))
sns.heatmap (df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# ### Dividing into X and Y sets for the model building

# In[34]:


y_train = df_train.pop('cnt')


# In[35]:


X_train = df_train


# In[36]:


X_train.info()


# 
# # Building a linear model

# In[37]:


## Recursive Feature Elimination
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# In[38]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# ### RFE - Recurssive Feature Elimination

# In[39]:


rfe = RFE(estimator=lm, n_features_to_select=12)
rfe = rfe.fit(X_train,y_train)


# In[40]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[41]:


col = list(X_train.columns[rfe.support_])
col


# In[42]:


X_train.columns[~rfe.support_]


# ### Build model iteratively. Using statsmodel for detailed statistics

# In[43]:


X_train_rfe = X_train[col]


# In[44]:


import statsmodels.api as sm
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train, X_train_rfe).fit()
print(lm.summary())


# In[45]:


X_train_new = X_train_rfe.drop (['const'], axis =1)
X_train_new.columns


# In[46]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X=X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Feature Elimination based on P Test and VIF Score

# In[47]:


# removing hum as it has high VIF
X_train_new2 = X_train_new.drop (['hum'], axis=1)


# In[48]:


X_train_new2.columns


# In[49]:


X_train_new2 = sm.add_constant(X_train_new2)

lm_2 = sm.OLS(y_train, X_train_new2).fit()
print(lm_2.summary())


# In[50]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
X=X_train_new2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[51]:


# removinh Holiday_Yes as high P value
X_train_new3 = X_train_new2.drop (['Holiday_Yes'], axis=1)


# In[52]:


X_train_new3.columns


# In[55]:


X_train_new3 = sm.add_constant(X_train_new3)

lm_3 = sm.OLS(y_train, X_train_new3).fit()
print(lm_3.summary())


# In[56]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
X=X_train_new3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# **Now we see VIFs and p-values are within an acceptable range. So we go ahead and make our predictions using this model only.**
# 
# #### R-squared train:                          0.829
# #### Adj. R-squared train:                  0.825 

# # Residual Analysis of the train data
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[57]:


y_train_predictbike = lm_3.predict(X_train_new3)


# In[58]:


res = y_train - y_train_predictbike
res


# **Distribution of the error terms**

# In[59]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_predictbike', fontsize = 15)         # X-label
plt.show()


# #### The residuals are following the normally distributed with a mean 0. All good!
# 

# #### Looking for patterns in the residuals

# In[60]:



plt.scatter(X_train.iloc[:, 0].values,res)
 
plt.show()


# In[61]:


fig = sm.qqplot(res, fit=True, line='45')
plt.show()


# # Making Predictions Using the Final Model
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final, i.e. 3rd model.
# 
# Applying the scaling on the test sets

# In[62]:


df_test


# In[63]:


df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[64]:


df_test.describe()


# In[65]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[66]:


# Adding constant variable to test dataframe
X_test_sm = sm.add_constant(X_test)
X_test_sm


# In[67]:


# Making predictions using the model

y_test_pred = lm_3.predict(X_test_sm[X_train_new3.columns])


# # Model Evaluation
# 

# In[68]:


from sklearn.metrics import r2_score


# In[69]:


r2_score(y_true = y_test, y_pred = y_test_pred)


# #### R-squared test:                          0.800

# Let's now plot the graph for actual versus predicted values.

# In[70]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
sns.regplot(y_test, y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)    
plt.show()


# #### Equation for best fitted line

# In[71]:


# list of coefs
lm_3.params


# 
# #### Equation:
# 
# $cnt = $0.469 * temp - $0.156 * windspeed - $0.081 * Spring +  $0.039 * Summer +  $0.078 * Winter -  $0.284 * LightSnow -  $0.0782 * MistCloudy + $0.0668 * Sat + $0.234 * Year_2019 + $0.05577 * Workingday_Yes
# 
# 

# # Conclusions
# 
# Model Evaluation looks fine. Overall we have decent model
# 
# Features affecting demand are below :
# - Temperature
# - Season
# - Working Day
# - Weather Situation 
# - Windspeed (inversely)
# 
# 
# - Demand is high during Working days
# - Demand is high during Summer seasons and less during Winter and Spring
# - Demand is high when weather is Clear and less during Mist Cloudy and Light Snow
# - Demand is high in 2019 compared to 2018
# - Demand increases with temperature
# - Demand decreases with windspeed
