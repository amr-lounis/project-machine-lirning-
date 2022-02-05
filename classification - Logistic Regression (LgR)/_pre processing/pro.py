import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pathcsv = './data_clean_one_vs_one_2vs4.csv'

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

# ------------------------------- predict
def predict_function(theta, X):
    probability = sigmoid_function(X * theta)
    return [1 if x >= 0.5 else 0 for x in probability]
    
def plotOnet(_theta,xName,yName):
    room_0 = data[data['room'].isin([0])]
    room_1 = data[data['room'].isin([1])]
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(room_0[xName], room_0[yName], s=10, c='r', marker='.', label='0')
    ax.scatter(room_1[xName], room_1[yName], s=10, c='b', marker='.', label='1')

    ax.legend()
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)

    _theta = np.hstack(np.array(_theta))
    x = np.linspace(data[xName].min(), data[xName].max(), 50)
    

    somTheta = 0
    for i in (1,X.shape[1]-1):
        somTheta = somTheta + _theta[i]  
    f =  theta[0]  -somTheta * x
    
    f =  theta[0] + theta[columns.index(xName)+1] * x + ( theta[columns.index(yName)+1] *  x)

    print(xName,columns.index(xName))    
    print(yName,columns.index(yName))
        
    
    f = np.hstack(np.array(f))
    
    ax.plot(x, f, 'r', label='p')
    ax.legend()
    
def plotAll(_theta):
    for i in range(0,len(columns)-1):
        for j in range(i+1,len(columns)-1):
            print(columns[i],columns[j])
            plotOnet(_theta,columns[i],columns[j])


print("-------------------------------") # ------------------------------- Plot
X,Y,data = redData(pathcsv)

def sigmoid_function(a):
    return 1.0 / (1 + np.exp(-a))

def cost_function(x, y, theta):
    m = x.shape[0]
    h = sigmoid_function(np.matmul(x, theta))
    cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1 -y.T), np.log(1 - h)))/m
    return cost

def gradient_Descent_function( _x , _y, _theta, _learning_rate):
    m = _x.shape[1]
    h = sigmoid_function(np.matmul(_x, _theta))
    grad = np.matmul(_x.T, (h - _y)) / m;
    _theta = _theta - _learning_rate * grad
    return _theta

def gradient_Descent_Iterations_function( _x , _y , _learning_rate, _repetition):
    _theta  = np.matrix(np.zeros(X.shape[1])).reshape([-1, 1])
    costs = np.zeros(_repetition)
    for i in range(n_iterations):
        _theta = gradient_Descent_function(_x, _y,_theta, _learning_rate)
        costs[i] = cost_function(_x, _y, _theta)
    return _theta , costs

def _plotCost(_costs):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(np.arange(len(_costs)), _costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training')
    
n_iterations  = 200
learning_rate = 0.01

theta ,costs =gradient_Descent_Iterations_function(X,Y,learning_rate,n_iterations)

print('theta of gradientDescent :',np.hstack(np.array(theta)))
print('cost  of gradientDescent :',costs[len(costs)-1])

p = predict_function(theta,X)
accuracy_function(Y, p)
#Cost
_plotCost(costs)
plotOnet(theta,'wifi_01','wifi_02')
# plotAll(theta)
