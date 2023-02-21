# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:25:59 2021

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Define Stacked RBF kernel function
def kr_RBF(X,Y,gamma=0.1):
    nn1=np.size(X,axis=0)
    nn2=np.size(Y,axis=0)
    dx=np.zeros((nn1,nn2))
    for i in range(0,nn1):
        for j in range(0,nn2):
            x1=X[i,:]
            x2=Y[j,:]
            dx[i,j]=np.sum(x1-x2)
    return np.exp(-0.5*(dx**2)/gamma)
    
# %% define Cross-information Kernel RBF function
def Xkr_RBF(X1,X2,Y1,Y2,gamma=0.1):
    nn1=np.size(X1,axis=0)
    nn2=np.size(X2,axis=0)
    dxx=np.zeros((nn1,nn2))
    dyy=np.zeros((nn1,nn2))
    dxy=np.zeros((nn1,nn2))
    dyx=np.zeros((nn1,nn2))
    for i in range(0,nn1):
        for j in range(0,nn2):
            x1=X1[i,:]
            x2=X2[j,:]
            y1=Y1[i,:]
            y2=Y2[j,:]
            dxx[i,j]=np.sum(x1-x2)
            dyy[i,j]=np.sum(y1-y2)
            dxy[i,j]=np.sum(x1-y2)
            dyx[i,j]=np.sum(y1-x2)
    return (2*np.exp(-0.5*(dxx**2)/gamma)+2*np.exp(-0.5*(dyy**2)/gamma)+
                    np.exp(-0.5*(dxy**2)/gamma)+np.exp(-0.5*(dyx**2)/gamma))


# %% main loop
a=0.1
sigma=1e-1
alpha=1e0

# %% Define data
d=1.8
c=a
ntransient=10000
n=ntransient+200

# %% logistic maps data
x=np.zeros(n)
y=np.zeros(n)
x[0]=1
y[0]=1
t=np.linspace(0,1,n)
for i in range(0,n-1):
    x[i+1]=1-d*x[i]**2
    y[i+1]=(1-c)*(1-d*y[i]**2)+c*(1-d*x[i]**2)
n=n-ntransient
# %% remove transient
x=x[ntransient:]
y=y[ntransient:]

# %% define time embeddings
p=1
xt=np.zeros((n,p))
yt=np.zeros((n,p))
for i in range(1,p+1):
    xt[i:,i-1]=x[0:-i]
    yt[i:,i-1]=y[0:-i]

# %% Define train test
ntrain = round(n*2/3)
Ytrain = yt[0:ntrain,:]
Ytest  = yt[ntrain:,:]
Xtrain = xt[0:ntrain,:]
Xtest  = xt[ntrain:,:]
ytrain = y[0:ntrain]
ytest  = y[ntrain:]


# %% AR model over y
#  Define kernel function and parameters
ktrain = kr_RBF(Ytrain,Ytrain,gamma=sigma)
ktest = kr_RBF(Ytest,Ytrain,gamma=sigma)
#  train using kernel ridge
w = (np.dot(np.linalg.inv(ktrain + 
                          alpha*np.identity(ntrain)), ytrain))
#  Predict using kernel ridge
yp_ar=np.dot(ktest,w)

# %% KGC
# %% Conditional AR model over y with x
#  Define kernel function and parameters  
ktrain = kr_RBF(np.concatenate((Xtrain,Ytrain),axis=1),
                np.concatenate((Xtrain,Ytrain),axis=1),gamma=sigma)
ktest = kr_RBF(np.concatenate((Xtest,Ytest),axis=1),
               np.concatenate((Xtrain,Ytrain),axis=1),gamma=sigma)
#  train using kernel ridge
w = np.dot(np.linalg.inv(ktrain + alpha*np.identity(ntrain)), ytrain)
#  Predict using kernel ridge
yp_kgc=np.dot(ktest,w)

# %% XKGC
# %% Conditional AR model over y with x
#  Define kernel function and parameters
ktrain = Xkr_RBF(Xtrain,Xtrain,Ytrain,Ytrain,gamma=sigma)
ktest = Xkr_RBF(Xtest,Xtrain,Ytest,Ytrain,gamma=sigma)
#  train using kernel ridge
w = np.dot(np.linalg.inv(ktrain + alpha*np.identity(ntrain)), ytrain)
#  Predict using kernel ridge
yp_xkgc=np.dot(ktest,w)

# %% plot results
plt.plot(ytest,c='k', label='test')
plt.plot(yp_ar,c='b', label='AR')
plt.plot(yp_kgc,c='g', label='KGC')
plt.plot(yp_xkgc,c='r', label='XKGC')
plt.grid()
plt.legend(loc="best",  scatterpoints=1, prop={'size': 12})
plt.show()