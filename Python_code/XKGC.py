# -*- coding: utf-8 -*-
"""

Explicit Kernel Granger Causality


Created on Dec 21 19:04 2020

@author: Diego Bueso Acevedo
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

import sys

# %% Definition of main functions
def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    
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


# Fixing random state for reproducibility
rng = np.random.RandomState(0)

# %% Define time series
# %% Define parameters
d=1.8               # Characteristic parameter of logistic function
c=0.1               # coupling strenght
ntransient=10000    # Number of time steps taken as a transition from intial conditions
nuseful=200         # Number of time stpes taken as useful data to test
n=ntransient+200    # Total number of time steps to run
# %% Create logistic maps time series
x=np.zeros(n)
y=np.zeros(n)
x[0]=1
y[0]=1
t=np.linspace(0,1,n)
for i in range(0,n-1):
    x[i+1]=1-d*x[i]**2
    y[i+1]=(1-c)*(1-d*y[i]**2)+c*(1-d*x[i]**2)

# %% remove transient
x=x[ntransient:]
y=y[ntransient:]
n=n-ntransient

# %% Definition test parameters
# Define the range of sigmas to test (RBF Kernel parameter, s-> inf = linear kernel)
ns=4  
sigma=np.logspace(-4,1,ns)  
# Define the range of regularization to test (l=0 not regalarized)                    
nl=6    
alpha=np.logspace(-6,1,nl)                    
# Define the number of cross-validations
nc=10                        

# Delta (Granger Causal index)
D_xkgc=np.zeros(nc)

# RMSe of each regression model to test
q_ar=np.zeros((nc,ns,nl))       # Autoregressive model
q_xkgc=np.zeros((nc,ns,nl))     # Cross-Kernel regression model

# %% main loop
for k in range(0,nc):
    
    
    # %% define time embeddings (min 1)
    p=1
    xt=np.zeros((n,p))
    yt=np.zeros((n,p))
    for i in range(1,p+1):
        xt[i:,i-1]=x[0:-i]
        yt[i:,i-1]=y[0:-i]
    
    # %% Define train test (with randomized cross-validation)
    idx=np.random.permutation(n)
    ytr=yt[idx,:]
    xtr=xt[idx,:]
    yr=y[idx]   
    
    ntrain = round(n*2/3)
    Ytrain = ytr[0:ntrain,:]
    Ytest  = ytr[ntrain:,:]
    Xtrain = xtr[0:ntrain,:]
    Xtest  = xtr[ntrain:,:]
    ytrain = yr[0:ntrain]
    ytest  = yr[ntrain:]
    
    # %% kernel parameter loop
    rms_ar=1
    rms_kgc=1
    rms_xkgc=1
    for s in range(0,ns):
        for l in range(0,nl):
            # %% AR model over y
            #  Define kernel function and parameters
            ktrain = kr_RBF(Ytrain,Ytrain,gamma=sigma[s])
            ktest = kr_RBF(Ytest,Ytrain,gamma=sigma[s])
            #  train using kernel ridge
            w = (np.dot(np.linalg.inv(ktrain + 
                                      alpha[l]*np.identity(ntrain)), ytrain))
            #  Predict using kernel ridge
            yp_ar=np.dot(ktest,w)
            #  result
            rms_aux_ar = sqrt(mean_squared_error(ytest, yp_ar))
            q_ar[k,s,l]=rms_aux_ar
            if rms_aux_ar<rms_ar:
                rms_ar=rms_aux_ar
        
            # %% XKGC
            # %% Conditional AR model over y with x
            #  Define kernel function and parameters
            ktrain = Xkr_RBF(Xtrain,Xtrain,Ytrain,Ytrain,gamma=sigma[s])
            ktest = Xkr_RBF(Xtest,Xtrain,Ytest,Ytrain,gamma=sigma[s])
            #  train using kernel ridge
            w = np.dot(np.linalg.inv(ktrain + alpha[l]*np.identity(ntrain)), ytrain)
            #  Predict using kernel ridge
            yp_xkgc=np.dot(ktest,w)
            #  result
            rms_aux_xkgc = sqrt(mean_squared_error(ytest, yp_xkgc))
            q_xkgc[k,s,l]=rms_aux_xkgc
            if rms_aux_xkgc<rms_xkgc:
                    rms_xkgc=rms_aux_xkgc;
    
    # %% Granger Causality index
    D_xkgc[k]=np.log(rms_ar/rms_xkgc)
    update_progress("Progress", k/(nc-1))

# %%  plot cross-validated Delta's
counts, bins = np.histogram(D_xkgc)
plt.hist(bins[:-1], bins, weights=counts)
plt.grid()
plt.xlabel('delta')
plt.title('x -> y')
plt.show()

# %% Mapping RMSe in function of sigma and regularization
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

im1=ax1.contourf(np.log10(alpha),np.log10(sigma),np.mean(q_ar,0),vmin = 0, vmax =0.6)
ax1.set_title('AR')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical');
ax1.grid()
ax1.set_ylabel('log10 Sigma (Kernel param.)')
ax1.set_xlabel('log10 Lambda (reg.)')
im1.set_cmap('viridis')

im2=ax2.contourf(np.log10(alpha),np.log10(sigma),np.mean(q_xkgc,0),vmin = 0, vmax =0.6)
ax2.set_title('XK')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical');
ax2.grid()
ax2.set_xlabel('log10 Lambda (reg.)')
im2.set_cmap('viridis')
plt.show()

# %%
a1=np.min(np.min(q_ar,2),1);
a2=np.min(np.min(q_xkgc,2),1);

# %% Mapping Delta in function of sigma and regularization
# Delta=np.log(np.mean(q_ar,0)/np.mean(q_xkgc,0));
# ax=plt.contourf(np.log10(alpha),np.log10(sigma),Delta,vmin = -np.max(np.abs(Delta)), vmax =np.max(np.abs(Delta)))
# plt.title('AR')
# plt.grid()
# plt.xlabel('log10 Sigma (Kernel param.)')
# plt.ylabel('log10 Lambda (reg.)')
# plt.colorbar();
# plt.set_cmap("PiYG")
# plt.show()
