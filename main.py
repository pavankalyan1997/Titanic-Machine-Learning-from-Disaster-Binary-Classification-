import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

traindataset=pd.read_csv('newtrain.csv')
X_train=traindataset.iloc[:,1:8].values
Y_train=traindataset.iloc[:,0:1].values.flatten()

testDataset=pd.read_csv('newtest.csv')
X_test=testDataset.iloc[:,:].values


#Taking care of missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X_train[:,2:3])
X_train[:,2:3]=imputer.transform(X_train[:,2:3])

imputerTest=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerTest=imputerTest.fit(X_test[:,2:3])
X_test[:,2:3]=imputerTest.transform(X_test[:,2:3])

imputerTest1=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerTest1=imputerTest1.fit(X_test[:,5:6])
X_test[:,5:6]=imputerTest.transform(X_test[:,5:6])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder1=LabelEncoder()
X_train[:,1]=labelencoder1.fit_transform(X_train[:,1])
X_test[:,1]=labelencoder1.fit_transform(X_test[:,1])





def encodeEmbarked(Embarked_dict,X):
    i=0
    count=-1
    for x in X[:,6]:
        if x not in Embarked_dict and type(x)!=float:
            Embarked_dict[x]=count+1
            count+=1
            X[:,6][i]=Embarked_dict[x]
        elif type(x)!=float:
            X[:,6][i]=Embarked_dict[x]
        i+=1
    return Embarked_dict,X


Embarked_dict={}
Embarked_dict,X_train=encodeEmbarked(Embarked_dict,X_train)
Embarked_dict,X_test=encodeEmbarked(Embarked_dict,X_test)

            
imputer1=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer1=imputer1.fit(X_train[:,6:7])
X_train[:,6:7]=imputer1.transform(X_train[:,6:7])
X_test[:,6:7]=imputer1.transform(X_test[:,6:7])

X_train = np.array(list(X_train[:,:]), dtype=np.float)
X_test = np.array(list(X_test[:,:]), dtype=np.float)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
scaler1=MinMaxScaler()
scaler1.fit(X_test)
X_test=scaler1.transform(X_test)


from sklearn.preprocessing import PolynomialFeatures
poly1=PolynomialFeatures(2)
X_train=poly1.fit_transform(X_train)
poly2=PolynomialFeatures(2)
X_test=poly2.fit_transform(X_test)



from LogisticRegression import LogisticRegression
classifier=LogisticRegression(X_train,Y_train)
J=classifier.computeCostFunction()
theta=classifier.returnTheta()
theta,Cost_history,theta_history=classifier.performGradientDescent(100000,0.01)
J=classifier.computeCostFunction()
Y_pred=classifier.predict(X_test)


solution=pd.read_csv('kaggle.csv') 
y_test=solution.iloc[:,1:2].values.flatten()

   
errorPercentage=(sum(abs(Y_pred-y_test)))/(len(Y_pred))*100
accuracy=100-errorPercentage


import csv
solutionData=[]
for i in range(len(Y_pred)):
    solutionData.append([892+i,Y_pred[i]])
    
with open('solution.csv','w',newline='') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerow(['PassengerId','Survived'])
    writer.writerows(solutionData)
csvFile.close()



    
        