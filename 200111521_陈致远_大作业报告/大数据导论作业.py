#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
#  R2 决定系数（r2_score）
from sklearn.metrics import r2_score
#MSE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[2]:


def load_split_data():
    train_data=pd.read_excel("数值化后的train.xlsx")
    print(train_data.head(1))
    y=train_data.values[:,-1]
    X=train_data.values[:,:-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    print("X_train.shape->",X_train.shape)
    print("y_train.shape->",y_train.shape)
    print("X_test.shape->",X_test.shape)
    print("y_test.shape->",y_test.shape)


# In[3]:


def train_model():
    model=RandomForestRegressor();
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)


# In[4]:


def model_evaluation():
    test_pred=pd.DataFrame()
    test_pred["y_true"]=y_test
    test_pred["y_pred"]=y_pred
    test_pred["y_error"]=np.abs(y_test-y_pred)
    test_pred["y_error_percentage"]=test_pred["y_error"]/test_pred["y_true"]
    test_pred.head(5)
    train_pred=pd.DataFrame()
    train_pred["y_true"]=y_train
    train_pred["y_pred"]=model.predict(X_train)
    train_pred["y_error"]=np.abs(train_pred["y_true"]-train_pred["y_pred"])
    train_pred["y_error_percentage"]=train_pred["y_error"]/train_pred["y_true"]
    train_pred.head(5)
    effct_test_dict = {}
    MSE = mean_squared_error(y_test, y_pred)
    MSE = round(MSE,3)
    MSE_list = []
    MSE_list.append(MSE)
    effct_test_dict["平均绝对误差(MSE)"] = MSE_list
    R2 = r2_score(y_test, y_pred)
    R2 = round(R2,3)
    R2_list = []
    R2_list.append(R2)
    effct_test_dict["R2 决定系数"] = R2_list
    RMSE = np.sqrt(MSE)
    RMSE = round(RMSE,3)
    RMSE_list = []
    RMSE_list.append(RMSE)
    effct_test_dict["均方根误差RMSE"] = RMSE_list
    print(effct_test_dict)
    effct_train_dict = {}
    MSE = mean_squared_error(y_train, model.predict(X_train))
    MSE = round(MSE,3)
    MSE_list = []
    MSE_list.append(MSE)
    effct_test_dict["平均绝对误差(MSE)"] = MSE_list
    R2 = r2_score(y_train, model.predict(X_train))
    R2 = round(R2,3)
    R2_list = []
    R2_list.append(R2)
    effct_test_dict["R2 决定系数"] = R2_list
    RMSE = np.sqrt(MSE)
    RMSE = round(RMSE,3)
    RMSE_list = []
    RMSE_list.append(RMSE)
    effct_test_dict["均方根误差RMSE"] = RMSE_list
    print(effct_test_dict)
    #在训练集上表示的十分良好，在测试集上表示效果相对较差


# In[5]:


def delete_data():
    #输出error_percentage里面最大的100个数，查看是否是由于异常数值的影响导致模型精度差
    import heapq
    tmp=zip(range(len(train_pred["y_error_percentage"])),train_pred["y_error_percentage"])
    key=heapq.nlargest(100,tmp,key=lambda x:x[1])
    # print(key)
    index=[];
    for item in key:
        index.append(item[0])
    print(index)
    train_data.drop(index=index,inplace=True)


# In[ ]:




