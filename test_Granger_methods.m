function [delta_linear,delta_stacked,delta_XK,surr_linear,surr_stacked,surr_XK]...
    =test_Granger_methods(y,x,p,nc,ns,sx,lamb)
N=size(y,1);
addpath([cd,'/surrogates']);

LtestAR_lin=zeros(nc,1);
LtestGC_lin=zeros(nc,1);

LtestAR=zeros(ns,nc);
LtestST=zeros(ns,nc);
LtestXK=zeros(ns,nc);

%% Build the time embeddings
YY = buffer(y,p+1,p)'; 
Y = YY(1:end,1:end-1); 
XX = buffer(x,p+1,p)'; 
X = XX(1:end,1:end-1); 

%% cross-validation
parfor ii=1:nc
Yt=Y;
Xt=X;
yt=y;
st=sx;

idx=randperm(N);
Ycros = Yt(idx,:);
Xcros = Xt(idx,:);
ycros = yt(idx);

%% define variables
Ntrain  = round(N*2/3);

Ytrain  = Ycros(1:Ntrain,:);
Ytest   = Ycros(Ntrain+1:end,:);
Xtrain  = Xcros(1:Ntrain,:);
Xtest   = Xcros(Ntrain+1:end,:);
ytrain  = ycros(1:Ntrain);
ytest   = ycros(Ntrain+1:end);

%% define kernel matrix
%AR
k=norm2mat(Ytest',Ytrain');
k=exp(-0.5*k);
KpredAR= @(s) k.^(1./s);
k=norm2mat(Ytrain',Ytrain');
k=exp(-0.5*k);
KtrainAR= @(s) k.^(1./s);
%stacked
Ztest=[Ytest,Xtest];
Ztrain=[Ytrain,Xtrain];
k=norm2mat(Ztest',Ztrain');
k=exp(-0.5*k);
KpredST= @(s) k.^(1./s);
k=norm2mat(Ztrain',Ztrain');
k=exp(-0.5*k);
KtrainST= @(s) k.^(1./s);

%cross-kernel
kyy=norm2mat(Ytest',Ytrain');
kxx=norm2mat(Xtest',Xtrain');
kyx=norm2mat(Ytest',Xtrain');
kxy=norm2mat(Xtest',Ytrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
kyx=exp(-0.5*kyx);
kxy=exp(-0.5*kxy);
KpredXK= @(s) 2*kyy.^(1./s)+2*kxx.^(1./s)+kyx.^(1./s)+kxy.^(1./s);
kyy=norm2mat(Ytrain',Ytrain');
kxx=norm2mat(Xtrain',Xtrain');
kyx=norm2mat(Ytrain',Xtrain');
kxy=norm2mat(Xtrain',Ytrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
kyx=exp(-0.5*kyx);
kxy=exp(-0.5*kxy);
KtrainXK= @(s) 2*kyy.^(1./s)+2*kxx.^(1./s)+kyx.^(1./s)+kxy.^(1./s);


%% define parametric lost fucntion
ypredAR = @(s) KpredAR(s)*((KtrainAR(s)+lamb*eye(Ntrain))\ytrain);
LAR   = @(s) rms(ytest-ypredAR(s));

ypredST = @(s) KpredST(s)*((KtrainST(s)+lamb*eye(Ntrain))\ytrain);
LST   = @(s) rms(ytest-ypredST(s));

ypredXK = @(s) KpredXK(s)*((KtrainXK(s)+lamb*eye(Ntrain))\ytrain);
LXK   = @(s) rms(ytest-ypredXK(s));

%% train Sigma
for i=1:ns
LtestAR(i,ii) = LAR(st(i));
LtestST(i,ii) = LST(st(i));
LtestXK(i,ii) = LXK(st(i));
end

%% linear GC test
ypred = Ytest*((Ytrain'*Ytrain+lamb*eye(p))\Ytrain'*ytrain);
LtestAR_lin(ii,1) = rms(ytest-ypred);
Ztest=[Ytest,Xtest];
Ztrain=[Ytrain,Xtrain];
ypred = Ztest*((Ztrain'*Ztrain+lamb*eye(2*p))\Ztrain'*ytrain);
LtestGC_lin(ii,1) = rms(ytest-ypred);

end

%% delta
dl=log(LtestAR_lin./LtestGC_lin);
ds=log(min(LtestAR,[],1)./min(LtestST,[],1));
dx=log(min(LtestAR,[],1)./min(LtestXK,[],1));

%% cross-validates mean
delta_linear   = nanmean(dl);
delta_stacked  = nanmean(ds);
delta_XK       = nanmean(dx);

%% optimun kernel parameter
[~,idx]=min(nanmean(LtestAR,2));
s_AR=sx(idx);
disp(idx);
[~,idx]=min(nanmean(LtestST,2));
s_stacked=sx(idx);
disp(idx);
[~,idx]=min(nanmean(LtestXK,2));
s_XK=sx(idx);
disp(idx);

%% surrogates
surr_linear    = surrogate_linear(x,y,0.05,lamb,p);
surr_stacked   = surrogate_stacked(x,y,0.05,s_AR,s_stacked,lamb,p);
surr_XK        = surrogate_XK(x,y,0.05,s_AR,s_XK,lamb,p);