#!/usr/bin/env python
# coding: utf-8

# ### import librrary

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read file

# In[2]:


pathcsv = '../_pre_processing/data_clean_one_vs_one_2vs4.csv'


# In[3]:


columns = [
    'wifi_01',
    'wifi_02',
    'wifi_03',
    'wifi_04',
    'wifi_05',
    'wifi_06',
    'wifi_07',
    'room'
    ]


# In[4]:


def redData(path):
    # read selected data
    data = pd.read_csv(path,usecols=columns)
    
    # rescaling data
    for i in range(0,len(columns)-1):
        data[columns[i]] = (data[columns[i]] - data[columns[i]].min()) / (data[columns[i]].max()-data[columns[i]].min())
    
    # convert data to numpy matrix
    cols = data.shape[1]
    xd = data.iloc[:,0:cols-1]
    yd = data.iloc[:,cols-1:cols] 
    x = np.matrix(xd.values)
    y = np.matrix(yd.values)
    
    ones = np.ones((x.shape[0],1))
    x = np.hstack([ones, x])
    
    return x,y,data


# In[5]:


def sigmoid_function(a):
    return 1.0 / (1 + np.exp(-a))


# In[6]:


def gradient_Descent_function( _x , _y, _theta, _learning_rate):
    m = _x.shape[1]
    h = sigmoid_function(np.matmul(_x, _theta))
    grad = np.matmul(_x.T, (h - _y)) / m;
    _theta = _theta - _learning_rate * grad
    return _theta


# In[7]:


def gradient_Descent_Iterations_function( _x , _y , _learning_rate, _repetition):
    _theta  = np.matrix(np.zeros(_x.shape[1])).reshape([-1, 1])
    costs = np.zeros(_repetition)
    for i in range(_repetition):
        _theta = gradient_Descent_function(_x, _y,_theta, _learning_rate)
        costs[i] = cost_function(_x, _y, _theta)
    return _theta , costs


# In[8]:


def cost_function(x, y, theta):
    m = x.shape[0]
    h = sigmoid_function(np.matmul(x, theta))
    cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1 -y.T), np.log(1 - h)))/m
    return cost


# In[9]:


# ------------------------------- predict
def predict_function(theta, X):
    probability = sigmoid_function(X * theta)
    return [1 if x >= 0.5 else 0 for x in probability]


# In[10]:


# ------------------------------- accuracy
def accuracy_function(_y,_p):
    sizeArray = len(_y)
    trueVal = 0
    errorVal = 0
    for i in range(0, sizeArray):
        if _y[i] == _p[i]:
            trueVal += 1
        else:
            errorVal +=1
    accuracy = trueVal*100 // sizeArray
    print ( 
        (
           'true  prediction {1}\n'+
           'wrong prediction {2}\n'+
           'total prediction {3}\n'+
           'accuracy = {0}% \n'
        ).format(accuracy,trueVal, errorVal ,sizeArray))


# In[11]:


def plot_cost(_costs):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(np.arange(len(_costs)), _costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs. Training')
    plt.show()


# In[12]:


def plotOne(_data,_theta,xName,yName):
    room_0 = _data[_data['room'].isin([0])]
    room_1 = _data[_data['room'].isin([1])]
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(room_0[xName], room_0[yName], s=10, c='r', marker='.', label='0')
    ax.scatter(room_1[xName], room_1[yName], s=10, c='b', marker='.', label='1')

    ax.legend()
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)

    _theta = np.hstack(np.array(_theta))
    x = np.linspace(_data[xName].min(), _data[xName].max(), 50)
    

    #somTheta = 0
    #for i in (1,X.shape[1]-1):
    #    somTheta = somTheta + _theta[i]  
    #f =  _theta[0] + somTheta * x/7
    
    f =  _theta[0] + ( _theta[columns.index(xName)+1] + _theta[columns.index(yName)+1])*x
    f = np.hstack(np.array(f))
    
    ax.plot(x, f, 'g', label='decision boundary')
    ax.legend()
    plt.show()


# In[13]:


def plotAll(_data,_theta):
    for i in range(0,len(columns)-1):
        for j in range(i+1,len(columns)-1):
            print(columns[i],columns[j])
            plotOne(_data,_theta,columns[i],columns[j])


# In[14]:


print("-------------------------------") # ------------------------------- Plot
X,Y,data = redData(pathcsv)
    
n_iterations  = 20
learning_rate = 0.1

theta ,costs =gradient_Descent_Iterations_function(X,Y,learning_rate,n_iterations)

print('theta of gradientDescent :',theta)
print('cost  of gradientDescent :',cost_function(X, Y, theta))

plot_cost(costs)


# In[15]:


p = predict_function(theta,X)
accuracy_function(Y, p)


# In[16]:


plotOne(data,theta,'wifi_01','wifi_04')
#plotAll(theta)


# In[ ]:





# In[ ]:




