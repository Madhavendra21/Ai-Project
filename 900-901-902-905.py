#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing the dataframe
diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df.head()


# In[2]:


diabetes_df.columns


# In[4]:


diabetes_df.info()


# In[5]:


diabetes_df.describe()


# In[6]:


# Know more about the dataset with transpose – here T is for the transpose
diabetes_df.describe().T


# In[7]:


#check if our dataset have null values or not
diabetes_df.isnull().head(10)


# In[8]:


#check the number of null values our dataset has
diabetes_df.isnull().sum()


# In[9]:



diabetes_df_copy = diabetes_df.copy(deep = True)
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Showing the Count of NANs
print(diabetes_df_copy.isnull().sum())

#we will be replacing the zeros with the NAN values so that we can impute it later to maintain the authenticity of the 
#dataset as well as trying to have a better Imputation approach i.e to apply mean values of each column to the null 
#values of the respective columns


# In[10]:


p = diabetes_df.hist(figsize = (20,20))


# In[11]:


# We will be imputing the mean value of the column to each missing value of that particular column
diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace = True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace = True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace = True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace = True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace = True)


# In[12]:


p = diabetes_df_copy.hist(figsize = (20,20))


# In[13]:


p = msno.bar(diabetes_df)


# In[14]:


color_wheel = {1: "#0392cf", 2: "#7bc043"}
colors = diabetes_df["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_df.Outcome.value_counts())
p=diabetes_df.Outcome.value_counts().plot(kind="bar")


# In[15]:


plt.subplot(121), sns.distplot(diabetes_df['Insulin'])
plt.subplot(122), diabetes_df['Insulin'].plot.box(figsize=(16,5))
plt.show()


# In[16]:


plt.figure(figsize=(12,10))
# seaborn has an easy method to showcase heatmap
p = sns.heatmap(diabetes_df.corr(), annot=True,cmap ='RdYlGn')


# ### Scaling the Data

# In[17]:


diabetes_df_copy.head()


# In[18]:


sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 
'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()


# In[19]:


y = diabetes_df_copy.Outcome
y


# ### Model Building

# In[20]:


X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']


# In[21]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,
                                                    random_state=7)


# In[22]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)


# In[23]:


rfc_train = rfc.predict(X_train)
from sklearn import metrics

print("Accuracy_Score =", format(metrics.accuracy_score(y_train, rfc_train)))


# In[24]:


from sklearn import metrics

predictions = rfc.predict(X_test)
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


pip install xgboost


# In[48]:


from xgboost import XGBClassifier

xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)


# In[34]:


from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)


# In[35]:


svc_pred = svc_model.predict(X_test)


# In[36]:


from sklearn import metrics

print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred)))


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test,svc_pred))


# In[38]:


rfc.feature_importances_


# In[39]:


(pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh'))


# In[40]:


import pickle

# Firstly we will be using the dump() function to save the model using pickle
saved_model = pickle.dumps(rfc)

# Then we will be loading that saved model
rfc_from_pickle = pickle.loads(saved_model)

# lastly, after loading that model we will use this to make predictions
rfc_from_pickle.predict(X_test)


# In[41]:


diabetes_df.head()


# In[42]:


diabetes_df.tail()


# In[43]:


rfc.predict([[0,137,40,35,168,43.1,2.228,33]]) #4th patient


# In[44]:


rfc.predict([[10,101,76,48,180,32.9,0.171,63]])  # 763 th patient


# In[ ]:




