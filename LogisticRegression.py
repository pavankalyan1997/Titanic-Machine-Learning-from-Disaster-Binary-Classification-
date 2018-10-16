import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

class LogisticRegression:
    def __init__(self,X,Y):
        #ones=np.ones(X.shape[0])
        #X=np.c_[ones,X]
        self.X=X
        self.Y=Y
        self.m=X.shape[0]
        self.n=X.shape[1]
        self.theta=np.random.randn(self.X.shape[1])
        

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    def computeCostFunction(self):
        h=self.sigmoid(np.matmul(self.X,self.theta))
        self.J=(-1/self.m)*np.sum(self.Y*np.log(h)+(1-self.Y)*np.log(1-h))
        return self.J
    
    def performGradientDescent(self,num_of_iter,alpha):
        self.Cost_history=[]
        self.theta_history=[]
        for x in range(num_of_iter):
            h=self.sigmoid(np.matmul(self.X,self.theta))
            J=self.computeCostFunction()
            self.Cost_history.append(J)
            self.theta_history.append(self.theta)
            temp=h-self.Y
            self.theta=self.theta-(alpha/self.m)*(self.X.T.dot(temp))
        return self.theta,self.Cost_history,self.theta_history
            
        
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
        
