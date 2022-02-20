import numpy as np
import pandas as pd
#Deviding the data set 
from sklearn.model_selection import train_test_split
# neural network
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score ,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

pathcsv = '../data/data_clean.csv'

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
    
    return x,y,data


X,y,data = redData(pathcsv)

X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
# print ("X_train.shape",X_train.shape)
# print ("Y_train.shape",y_train.shape)
# print ("X_test.shape",X_test.shape)
# print ("Y_test.shape",y_test.shape)


_alpha =0.1 #2
_hiddenLS = (7, 6)
# clf = MLPClassifier(solver='lbfgs', alpha=_alpha, hidden_layer_sizes=_hiddenLS, random_state=1,verbose=False)
# clf.fit( X_train, y_train )

clf = MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=_hiddenLS, random_state=1)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
print("Predection : /n", prediction.T)
print("Class y : /n",  y_test.T)
#-----------------------------------------------------------------------------
c_m =confusion_matrix(y_test,prediction)
print('confusion matrix :\n',c_m,"\n")
#-----------------------------------------------------------------------------
p_s= metrics.accuracy_score(prediction,y_test)
print('The accuracy :',p_s,"\n")
#-----------------------------------------------------------------------------
p_s= precision_score(y_test, prediction,pos_label='positive',average='micro')
print('presicion score : {}'.format(p_s),"\n")
#-----------------------------------------------------------------------------
r_s=recall_score(y_test, prediction,pos_label='positive',average='micro')
print('recall score : {}'.format(r_s),"\n")
#-----------------------------------------------------------------------------
f1_s= f1_score(y_test, prediction ,pos_label='positive',average='micro')
print('f1 score : {}'.format(f1_s),"\n")
#-----------------------------------------------------------------------------
df_cm = pd.DataFrame(c_m,  index = [i for i in range(0,4)], columns = [i for i in range(0,4)])
plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('Neural Network \nAccuracy:{0:.3f}'.format(metrics.accuracy_score(y_test, prediction)))
plt.ylabel('True label')
plt.xlabel('Predicted label')


print(classification_report(y_test,prediction,target_names=['unaccepted','accepted','good','verygood']))

