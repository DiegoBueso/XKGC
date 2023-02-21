function Fc=surrogate_XK(x,y,alpha,s_AR,s_XK,lamb,p)
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

%cross-kernel
kyy=norm2mat(Ytest',Ytrain');
kxx=norm2mat(Xtest',Xtrain');
kyx=norm2mat(Ytest',Xtrain');
kxy=norm2mat(Xtest',Ytrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
kyx=exp(-0.5*kyx);
kxy=exp(-0.5*kxy);
Kpred_XK= 2*kyy.^(1./s_XK)+2*kxx.^(1./s_XK)+kyx.^(1./s_XK)+kxy.^(1./s_XK);
kyy=norm2mat(Ytrain',Ytrain');
kxx=norm2mat(Xtrain',Xtrain');
kyx=norm2mat(Ytrain',Xtrain');
kxy=norm2mat(Xtrain',Ytrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
kyx=exp(-0.5*kyx);
kxy=exp(-0.5*kxy);
Ktrain_XK= 2*kyy.^(1./s_XK)+2*kxx.^(1./s_XK)+kyx.^(1./s_XK)+kxy.^(1./s_XK);
ypred_XK =  Kpred_XK*((Ktrain_XK+lamb*eye(Ntrain))\ytrain);
L_XK   = rms(ytest-ypred_XK);

%% delta
Fc(i)=log(L_AR/L_XK);
end
Fc=max(Fc);
