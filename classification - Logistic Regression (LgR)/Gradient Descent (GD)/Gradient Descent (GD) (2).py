#!/usr/bin/env python
# coding: utf-8

# ### import librrary

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### select columns

# In[2]:


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


# ### function  read data and convert to matrix

# In[3]:


def redData(path):
    # read selected data
    data = pd.read_csv(path,usecols=columns)
    
    # convert data to numpy matrix
    cols = data.shape[1]
    xd = data.iloc[:,0:cols-1]
    yd = data.iloc[:,cols-1:cols] 
    x = np.matrix(xd.values)
    y = np.matrix(yd.values)
    
    # inset column One for theta[0]
    ones = np.ones((x.shape[0],1))
    x = np.hstack([ones, x])
    
    return x,y,data


# ### function hypothesis function return probability between 1 and 0

# In[4]:


def hypothesis_function(a):
    return 1.0 / (1 + np.exp(-a))


# ### function  cost

# In[5]:


def cost_function(x, y, theta):
    m = x.shape[0]
    h = hypothesis_function(np.matmul(x, theta))
    cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1 -y.T), np.log(1 - h)))/m
    return cost


# ### function  J

# In[6]:


def gradient_Descent_function( _x , _y, _theta, _learning_rate):
    m = _x.shape[1]
    h = hypothesis_function(np.matmul(_x, _theta))
    grad = np.matmul(_x.T, (h - _y)) / m;
    _theta = _theta - _learning_rate * grad
    return _theta


# ### return theta of Gradient Descent binary classifiers of minimall cost and array of all cost

# In[7]:


def gradient_Descent_Iterations_function( _x , _y , _learning_rate, _repetition):
    _theta  = np.matrix(np.zeros(_x.shape[1])).reshape([-1, 1])
    costs = np.zeros(_repetition)
    _t = []
    for i in range(_repetition):
        _theta = gradient_Descent_function(_x, _y,_theta, _learning_rate)
        _t.append(_theta)
        costs[i] = cost_function(_x, _y, _theta)
    index_min_cost = np.argmin(costs)
    print("nomber Iterations of minimal cost is ",index_min_cost)
    return _t[index_min_cost] , costs


# ### function predict

# In[8]:


def predict_function(theta, X):
    probability = hypothesis_function(X * theta)
    return [1 if x >= 0.5 else 0 for x in probability]


# ### function accuracy

# In[9]:


def accuracy_function(_y,_p):
    sizeArray = len(_y)
    trueVal = 0
    errorVal = 0
    for i in range(0, sizeArray):
        if _y[i] == _p[i]:
            trueVal += 1
        else:
            errorVal +=1
    accuracy = trueVal*100 / sizeArray
    print ( 
        (
           'true  prediction {1}\n'+
           'wrong prediction {2}\n'+
           'total prediction {3}\n'+
           'accuracy = {0:5.2f}% \n'
        ).format(accuracy,trueVal, errorVal ,sizeArray))


# ### function plot costs of trining

# In[10]:


def plot_cost(_costs):
    index_min_cost = np.argmin(_costs)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(np.arange(len(_costs)), _costs, 'r')
    ax.axvline(x=index_min_cost, color='b', linestyle='-',);
    plt.text(index_min_cost+0.3,2,'minimal cost',rotation=90)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs. Training')
    plt.show()

def plot_hypothesis_function():
    x = np.linspace(-20, 20, 400)
    plt.figure(1)
    plt.plot(x, hypothesis_function(x))
    plt.axvline(x=0, color='r', linestyle='-');
    plt.title("hypothesis");
    
# ### plot data

# In[11]:


def plotAll(_data,xName,yName):
    room_1 = _data[_data['room'].isin([1])]
    room_2 = _data[_data['room'].isin([2])]
    room_3 = _data[_data['room'].isin([3])]
    room_4 = _data[_data['room'].isin([4])]
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(room_1[xName], room_1[yName], s=10, c='r', marker='.', label='room_1')
    ax.scatter(room_2[xName], room_2[yName], s=10, c='g', marker='.', label='room_2')
    ax.scatter(room_3[xName], room_3[yName], s=10, c='b', marker='.', label='room_3')
    ax.scatter(room_4[xName], room_4[yName], s=10, c='y', marker='.', label='room_4')

    ax.legend()
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)

    plt.show()  


# ### function run trining and return theta 

# In[12]:


paths_ovr= ['../data/data_clean_one_vs_rest_1.csv',
            '../data/data_clean_one_vs_rest_2.csv',
            '../data/data_clean_one_vs_rest_3.csv',
            '../data/data_clean_one_vs_rest_4.csv']

def run_trining(_path,_learning_rate,_n_iterations):
    X,Y,data = redData(_path)
    theta ,costs =gradient_Descent_Iterations_function(X,Y,_learning_rate,_n_iterations)
    print('-- theta of gradientDescent :',_path,'\n',theta.T)
    print('cost  of gradientDescent :',cost_function(X, Y, theta))
    # plot cost vs Training
    plot_cost(costs)
    # plot cost vs Training
    p = predict_function(theta,X)
    accuracy_function(Y, p)
    return theta


# ### trining model room 1 vs Rest

# In[13]:


theta_room1 = run_trining(paths_ovr[0],0.08,100)


# ### trining model room 2 vs Rest

# In[ ]:


theta_room2 = run_trining(paths_ovr[1],0.08,100)


# ### trining model room 3 vs Rest

# In[ ]:


theta_room3 = run_trining(paths_ovr[2],0.08,100)


# ### trining model room 4 vs Rest

# In[ ]:


theta_room4 = run_trining(paths_ovr[3],0.08,100)


# # multi class classification
# 

# ### all theta in one array

# In[ ]:


theta_all = [theta_room1,theta_room2,theta_room3,theta_room4,]


# ### function predict multiClass for new one X

# In[ ]:


def predict_multiClass(_theta_all,_x_new):
    _probabilities = []
    for i in range(0,len(_theta_all)):
        _probability = hypothesis_function(_x_new * _theta_all[i])
        _probabilities.append(_probability)
    _index_of_max = np.argmax(_probabilities)
    # print('probabilities : ',_probabilities,)
    # print('max probability in index: ',_index_of_max,)
    return _index_of_max + 1 # number of room equel index +1


# ### testing multi class classification for new x

# In[ ]:


#test for multi class classification all X in dataset
theta_all = [theta_room1,theta_room2,theta_room3,theta_room4,]

X,Y,data = redData('../data/data_clean.csv')

vv= data.corr()
y_predict_rooms =[]
for v in X:
    p=predict_multiClass(theta_all,v)
    y_predict_rooms.append(p)

accuracy_function(Y,y_predict_rooms)

plotAll(data,'wifi_01','wifi_05',)
plot_hypothesis_function()

