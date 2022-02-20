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
    
    # inset column One for theta[0]
    ones = np.ones((x.shape[0],1))
    x = np.hstack([ones, x])
    
    return x,y,data

#Importing the dataset
X,y,data = redData('../data/data_clean.csv')

# X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
# print ("X_train.shape",X_train.shape)
# print ("Y_train.shape",y_train.shape)
# print ("X_test.shape",X_test.shape)
# print ("Y_test.shape",y_test.shape)

# 60/20/20 (train/cv/test) data split
data = np.array(data)
np.random.seed(5)
np.random.shuffle(data)
num_train = int(.6 * len(data))
num_cv = num_train + int(.2 * len(data))
x_train, y_train = data[:num_train, :-1], data[:num_train, -1]
x_cv, y_cv = data[num_train:num_cv, :-1], data[num_train:num_cv, -1]
x_test, y_test = data[num_cv:, :-1], data[num_cv:, -1]


print(f'm = {x_train.shape[0]}\nn = {x_train.shape[1]}')

svc = svm.SVC(C=0.1, kernel='rbf', gamma=0.1)
svc.fit(x_train, y_train)

print('Train accuracy = {0}%'.format(np.round(svc.score(x_train, y_train) * 100, 2)))
print('CV accuracy = {0}%'.format(np.round(svc.score(x_cv, y_cv) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(x_test, y_test) * 100, 2)))

