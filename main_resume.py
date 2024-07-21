#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Load the data
df = pd.read_csv("UpdatedResumeDataSet.csv")
df.head()


# In[4]:


df.shape


# In[10]:


# Ensure 'Category' is treated as categorical data
df['Category'] = df['Category'].astype('category')


# In[6]:


df['Category'].value_counts()


# In[12]:


# Create the count plot
plt.figure(figsize=(20, 5))
sns.countplot(x='Category', data=df)
plt.xticks(rotation=90)  # Rotate the x labels if they are too long
plt.show()


# In[14]:


count = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize=(20, 5))
plt.pie(count,labels=labels,autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()


# In[15]:


#Data Preprocessing
df['Resume'][0]


# In[16]:


#to remove urls, hashtags, mentions, special letters and puntuations
import re

def preprocess_data(text):
    
    preprocess_txt = re.sub('http\S+\s',' ',text)
    
    preprocess_txt = re.sub('@\S+',' ',preprocess_txt)
    
    preprocess_txt = re.sub('#\S+\s',' ',preprocess_txt)
    
    preprocess_txt = re.sub('RT|cc',' ',preprocess_txt)
    
    preprocess_txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',preprocess_txt)
    
    preprocess_txt = re.sub(r'[^\x00-\x7f]',' ',preprocess_txt)
    
    preprocess_txt = re.sub('\s+',' ',preprocess_txt)
    
    return preprocess_txt


# In[18]:


df['Resume'] = df['Resume'].apply(lambda x: preprocess_data(x))


# In[19]:


df['Resume']


# In[39]:


df['Resume'][0]


# In[21]:


#words into categorical value

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(df['Category'])

df['Category'] = le.transform(df['Category'])


# In[23]:


df['Category'].unique()


# In[26]:


#vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(stop_words='english')

tf.fit(df['Resume'])

transformed_txt = tf.transform(df['Resume'])


# In[29]:


#split and train data 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(transformed_txt,df['Category'],random_state=42,test_size=0.3)


# In[30]:


x_train.shape


# In[31]:


x_test.shape


# In[34]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

kn = OneVsRestClassifier(KNeighborsClassifier())

kn.fit(x_train,y_train)

y_pred = kn.predict(x_test)

print(y_pred)


# In[35]:


#Evaluation of knn model

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# In[37]:


#to load trained classifier

import pickle

pickle.dump(tf,open('tf.pkl','wb'))
pickle.dump(kn,open('kn.pkl','wb'))


# In[40]:


#to load trained classifier

import pickle
kn_pkl = pickle.load(open('kn.pkl','rb'))


# In[ ]:




