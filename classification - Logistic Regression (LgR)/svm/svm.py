#Importing the libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

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
    
    # convert data to numpy matrix
    cols = data.shape[1]
    xd = data.iloc[:,0:cols-1]
    yd = data.iloc[:,cols-1:cols] 
    x = np.matrix(xd.values)
    y = np.matrix(yd.values)

    return x,y,data

#Importing the dataset
X,y,data = redData('../data/data_clean.csv')

x_train,  x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
print ("X_train.shape",x_train.shape)
print ("Y_train.shape",y_train.shape)
print ("X_test.shape",x_test.shape)
print ("Y_test.shape",y_test.shape)

print(f'm = {x_train.shape[0]}\nn = {x_train.shape[1]}')

svc = svm.SVC(C=0.1, kernel='rbf', gamma=0.1)
svc.fit(x_train, y_train)

print('Train accuracy = {0}%'.format(np.round(svc.score(x_train, y_train) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(x_test, y_test) * 100, 2)))

