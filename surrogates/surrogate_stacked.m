function Fc=surrogate_stacked(x,y,alpha,s_AR,s_stacked,lamb,p)
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

%AR
k=norm2mat(Ytest',Ytrain');
Kpred_AR=exp(-0.5*k./s_AR);
k=norm2mat(Ytrain',Ytrain');
Ktrain_AR=exp(-0.5*k./s_AR);
ypred_AR =  Kpred_AR*((Ktrain_AR+lamb*eye(Ntrain))\ytrain);
L_AR   =  rms(ytest-ypred_AR);

%stacked
Ztest=[Ytest,Xtest];
Ztrain=[Ytrain,Xtrain];
k=norm2mat(Ztest',Ztrain');
Kpred_stacked=exp(-0.5*k./s_stacked);
k=norm2mat(Ztrain',Ztrain');
Ktrain_stacked=exp(-0.5*k./s_stacked);
ypred_stacked =  Kpred_stacked*((Ktrain_stacked+lamb*eye(Ntrain))\ytrain);
L_stacked   = rms(ytest-ypred_stacked);

%% delta
Fc(i)=log(L_AR/L_stacked);
end
Fc=max(Fc);
