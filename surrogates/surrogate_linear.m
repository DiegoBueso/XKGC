function Fc=surrogate_linear(x,y,alpha,lamb,p)
d=round(1/alpha-1);
Fc=zeros(d,1);
for i=1:d
xr=generate_time_sift(x);
yr=generate_time_sift(y);
N=size(xr,1);

%% Build the time embeddings
YY = buffer(yr,p+1,p)'; 
XX = buffer(xr,p+1,p)';
Y = YY(1:end,1:end-1); 
X = XX(1:end,1:end-1);
idx=randperm(N);
Y = Y(idx,:);
X = X(idx,:);
yr = yr(idx);

%% define variables
Ntrain  = round(N*2/3);

Ytrain  = Y(1:Ntrain,:);
Ytest   = Y(Ntrain+1:end,:);
Xtrain  = X(1:Ntrain,:);
Xtest   = X(Ntrain+1:end,:);
ytrain  = yr(1:Ntrain);
ytest   = yr(Ntrain+1:end);

%% linear GC test
ypred = Ytest*((Ytrain'*Ytrain+lamb*eye(p))\Ytrain'*ytrain);
L_AR_lin = rms(ytest-ypred);
Ztest=[Ytest,Xtest];
Ztrain=[Ytrain,Xtrain];
ypred = Ztest*((Ztrain'*Ztrain+lamb*eye(2*p))\Ztrain'*ytrain);
L_GC_lin = rms(ytest-ypred);

Fc(i)=log(L_AR_lin/L_GC_lin);
end
Fc=max(Fc);
