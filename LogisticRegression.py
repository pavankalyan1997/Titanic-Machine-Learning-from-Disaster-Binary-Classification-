import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

class LogisticRegression:
    def __init__(self,X,Y,X_val,Y_val,X_test):
        #ones=np.ones(X.shape[0])
        #X=np.c_[ones,X]
        self.X=X
        self.Y=Y
        self.X_val=X_val
        self.Y_val=Y_val
        self.X_test=X_test
        self.m=X.shape[0]
        self.n=X.shape[1]
        self.theta=np.random.randn(self.X.shape[1])
        

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    def computeCostFunction(self,Lambda):
        h=self.sigmoid(np.matmul(self.X,self.theta))
        self.J=(-1/self.m)*np.sum(self.Y*np.log(h)+(1-self.Y)*np.log(1-h))
        temp=self.theta
        temp[0]=0
        Reg=Lambda*np.sum(self.theta**2)
        self.J+=Reg
        return self.J
    
    def performGradientDescent(self,num_of_iter,alpha,Lambda):
        self.Cost_history=[]
        self.theta_history=[]
        self.TrainingError=[]
        self.ValError=[]
        for x in range(num_of_iter):
            h=self.sigmoid(np.matmul(self.X,self.theta))
            hVal=self.sigmoid(np.matmul(self.X_val,self.theta))
            trainError=self.computeError(h,self.Y)
            valError=self.computeError(hVal,self.Y_val)
            self.TrainingError.append(trainError)
            self.ValError.append(valError)
            J=self.computeCostFunction(Lambda)
            self.Cost_history.append(J)
            RegTheta=self.theta
            RegTheta[0]=0
            self.theta_history.append(self.theta)
            temp=h-self.Y
            self.theta=self.theta-(alpha/self.m)*(self.X.T.dot(temp))-((alpha*Lambda)/self.m)*RegTheta
        return self.theta,self.Cost_history,self.theta_history,self.TrainingError,self.ValError
    
    def computeError(self,h,y):
        m=len(y)
        return (1/2/m)*np.sum((h-y)**2)
            
        
    def predict(self,X_test):
        #ones=np.ones(X_test.shape[0])
        #X_test=np.c_[ones,X_test]
        self.Y_pred=np.matmul(X_test,self.theta)
        for i in range(len(self.Y_pred)):
            if self.Y_pred[i]>0:
                self.Y_pred[i]=1
            else: self.Y_pred[i]=0
        
        return self.Y_pred
    
        
    def returnTheta(self):
        return self.theta
    
    def returnX(self):
        return self.X
        
    def returnY(self):
        return self.Y
        
