
# coding: utf-8

# In[1]:

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import scipy
import xgboost as xgb


# In[2]:

#Importing the Data 
train =  pd.read_csv("D:\\Kaggle Projects\\Market Check\\train.csv")
test =  pd.read_csv("D:\\Kaggle Projects\\Market Check\\test.csv")
test_l =  pd.read_csv("D:\\Kaggle Projects\\Market Check\\test.csv")


# In[3]:

train.head(10)


# In[4]:

#Updating the Object Values for Train Data 
train.loc[(train.City_Category== 'B') ,'City_Category' ] = 1
train.loc[(train.City_Category== 'A') ,'City_Category' ] = 3
train.loc[(train.City_Category== 'C') ,'City_Category' ] = 2
train.loc[(train.Age== '0-17') ,'Age' ] = 1
train.loc[(train.Age== '18-25') ,'Age' ] = 2
train.loc[(train.Age== '26-35') ,'Age' ] = 3
train.loc[(train.Age== '36-45') ,'Age' ] = 4
train.loc[(train.Age== '46-50') ,'Age' ] = 5
train.loc[(train.Age==  '51-55') ,'Age' ] = 6
train.loc[(train.Age== '55+') ,'Age' ] = 7
train.loc[(train.Gender== 'F') ,'Gender' ] = 1
train.loc[(train.Gender== 'M') ,'Gender' ] = 2
train.loc[(train.Stay_In_Current_City_Years == '0') , 'Stay_In_Current_City_Years'] = 0
train.loc[(train.Stay_In_Current_City_Years == '1') , 'Stay_In_Current_City_Years'] = 1
train.loc[(train.Stay_In_Current_City_Years == '2') , 'Stay_In_Current_City_Years'] = 2
train.loc[(train.Stay_In_Current_City_Years == '3') , 'Stay_In_Current_City_Years'] = 3
train.loc[(train.Stay_In_Current_City_Years == '4+') , 'Stay_In_Current_City_Years'] = 4


# In[5]:

#Updating the Object Values for Test Data 
test.loc[(test.City_Category== 'B') ,'City_Category' ] = 1
test.loc[(test.City_Category== 'A') ,'City_Category' ] = 3
test.loc[(test.City_Category== 'C') ,'City_Category' ] = 2
test.loc[(test.Age== '0-17') ,'Age' ] = 1
test.loc[(test.Age== '18-25') ,'Age' ] = 2
test.loc[(test.Age== '26-35') ,'Age' ] = 3
test.loc[(test.Age== '36-45') ,'Age' ] = 4
test.loc[(test.Age== '46-50') ,'Age' ] = 5
test.loc[(test.Age==  '51-55') ,'Age' ] = 6
test.loc[(test.Age== '55+') ,'Age' ] = 7
test.loc[(test.Gender== 'F') ,'Gender' ] = 1
test.loc[(test.Gender== 'M') ,'Gender' ] = 2
test.loc[(test.Stay_In_Current_City_Years == '0') , 'Stay_In_Current_City_Years'] = 0
test.loc[(test.Stay_In_Current_City_Years == '1') , 'Stay_In_Current_City_Years'] = 1
test.loc[(test.Stay_In_Current_City_Years == '2') , 'Stay_In_Current_City_Years'] = 2
test.loc[(test.Stay_In_Current_City_Years == '3') , 'Stay_In_Current_City_Years'] = 3
test.loc[(test.Stay_In_Current_City_Years == '4+') , 'Stay_In_Current_City_Years'] = 4


# In[6]:

train.head(5)


# In[7]:

test.head(5)


# In[8]:

train[['Gender','Age','Stay_In_Current_City_Years','City_Category']] = train[['Gender','Age','Stay_In_Current_City_Years','City_Category']].apply(pd.to_numeric)
test[['Gender','Age','Stay_In_Current_City_Years','City_Category']] = test[['Gender','Age','Stay_In_Current_City_Years','City_Category']].apply(pd.to_numeric)


# In[9]:

train['Product_ID'] = train['Product_ID'].map(lambda x: x.lstrip('P'))
test['Product_ID'] = test['Product_ID'].map(lambda x: x.lstrip('P'))


# In[10]:

train['Product_ID'] = train['Product_ID'].apply(pd.to_numeric)
test['Product_ID'] = test['Product_ID'].apply(pd.to_numeric)


# In[11]:

train.info()


# In[12]:

train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)


# In[13]:

train.describe()


# In[14]:

test.describe()


# In[15]:

y_train = train['Purchase']


# In[16]:

train_new = train.drop('Purchase', 1)


# In[17]:

train_new.head(5)


# In[18]:

train_new.size


# In[19]:

train_X = np.array(train_new).astype('float')
test_X = np.array(test).astype('float')


# In[20]:

train_y = np.array(y_train)


# In[21]:

xgtrain = xgb.DMatrix(train_X, label=train_y)


# In[22]:

xgtest = xgb.DMatrix(test_X)


# In[23]:

params = {}


# In[29]:

params["objective"] = "reg:linear"
params["eta"] = 0.05
params["max_depth"] = 10
params["seed"] = 0
plst = list(params.items())
num_rounds = 700


# In[31]:

#Why Using these parameters 
#When you observe high training accuracy, but low tests accuracy, it is likely that you encounter overfitting problem.
# As to control that we have directly control model complexity by defining parameters to the model
# max_depth
# the second way is to add randomness 
#You can also reduce stepsize eta, but needs to remember to increase num_round when you do so thats why we have used 
# ets=0.05 and number of rounds as 700
#type of regressor used is linear


# In[32]:

model = xgb.train(plst, xgtrain, num_rounds)


# In[33]:

pred_test_y = model.predict(xgtest)


# In[34]:

test_user_id = np.array(test_l["User_ID"])
test_product_id = np.array(test_l["Product_ID"])


# In[36]:

out_df = pd.DataFrame({"User_ID":test_user_id})
out_df["Product_ID"] = test_product_id
out_df["Purchase"] = pred_test_y
out_df.to_csv("D:\Kaggle Projects\Market Check\\final_submission.csv", index=False)


# In[ ]:



