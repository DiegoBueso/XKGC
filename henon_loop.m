clc; close all; clear;


%% parameter
np=100;
nc=12;
ns=50;
sx=logspace(-10,6,ns)';
lamb=1E-12;
N=3000;
Ntran=2900;

%% outputs
LtestAR=zeros(ns,nc);
LtestCA=zeros(ns,nc);
LtestCAS=zeros(ns,nc);
LtestCAX=zeros(ns,nc);

D=zeros(np,1);
Ds=zeros(np,1);
Dx=zeros(np,1);

Lar=zeros(np,1);
Lca=zeros(np,1);
Lcs=zeros(np,1);
Lcx=zeros(np,1);

C=linspace(0,1,np)';
for j=1:np
%% logistic model
N=3000;
c=C(j);
x = ones(N,1);
y = ones(N,1);
for i=3:N
        x(i) = 1.4 - x(i-1).^2 +0.3*x(i-2);
        y(i) = 1.4 -(1-c)*y(i-1).^2 -c*x(i-1)*y(i-1)+0.3*y(i-2);
end


%% eliminate transient
x=x(Ntran+1:end);
y=y(Ntran+1:end);
N=N-Ntran;

%% Build the time embeddings
p= 2;
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

%% kernels
%AR
k=norm2mat(Ytest',Ytrain');
k=exp(-0.5*k);
KpredAR= @(s) k.^(1./s);
k=norm2mat(Ytrain',Ytrain');
k=exp(-0.5*k);
KtrainAR= @(s) k.^(1./s);

%stack
Ztest=[Ytest,Xtest];
Ztrain=[Ytrain,Xtrain];
k=norm2mat(Ztest',Ztrain');
k=exp(-0.5*k);
KpredCA= @(s) k.^(1./s);
k=norm2mat(Ztrain',Ztrain');
k=exp(-0.5*k);
KtrainCA= @(s) k.^(1./s);

%Sumation
kyy=norm2mat(Ytest',Ytrain');
kxx=norm2mat(Xtest',Xtrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
KpredCAS= @(s) kyy.^(1./s)+kxx.^(1./s)
kyy=norm2mat(Ytrain',Ytrain');
kxx=norm2mat(Xtrain',Xtrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
KtrainCAS= @(s) kyy.^(1./s)+kxx.^(1./s);

%cross-kernel
kyy=norm2mat(Ytest',Ytrain');
kxx=norm2mat(Xtest',Xtrain');
kyx=norm2mat(Ytest',Xtrain');
kxy=norm2mat(Xtest',Ytrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
kyx=exp(-0.5*kyx);
kxy=exp(-0.5*kxy);
KpredCAX= @(s) 2*kyy.^(1./s)+2.*kxx.^(1./s)+kyx.^(1./s)+kxy.^(1./s);
% KpredCAX= @(s) kyy.^(1./s).*kxx.^(1./s)+2*kyy.^(1./s)+2.*kxx.^(1./s)+kyx.^(1./s)+kxy.^(1./s);
kyy=norm2mat(Ytrain',Ytrain');
kxx=norm2mat(Xtrain',Xtrain');
kyx=norm2mat(Ytrain',Xtrain');
kxy=norm2mat(Xtrain',Ytrain');
kyy=exp(-0.5*kyy);
kxx=exp(-0.5*kxx);
kyx=exp(-0.5*kyx);
kxy=exp(-0.5*kxy);
KtrainCAX= @(s) 2*kyy.^(1./s)+2.*kxx.^(1./s)+kyx.^(1./s)+kxy.^(1./s);
% KtrainCAX= @(s) kyy.^(1./s).*kxx.^(1./s)+2*kyy.^(1./s)+2.*kxx.^(1./s)+kyx.^(1./s)+kxy.^(1./s);
%% train Sigma
for i=1:ns
w=0.01;

Kp=KpredAR(st(i));Kt=KtrainAR(st(i));
ypred =Kp*((Kt+lamb*eye(Ntrain))\ytrain);
LtestAR(i,ii)=rms(ytest-ypred);

Kp=KpredCA(st(i));Kt=KtrainCA(st(i));
ypred =Kp*((Kt+lamb*eye(Ntrain))\ytrain);
LtestCA(i,ii)=rms(ytest-ypred);

Kp=KpredCAS(st(i));Kt=KtrainCAS(st(i));
ypred =Kp*((Kt+lamb*eye(Ntrain))\ytrain);
LtestCAS(i,ii)=rms(ytest-ypred);

Kp=w*KpredCA(st(i))+(1-w)*KpredCAX(st(i));Kt=w*KtrainCA(st(i))+(1-w)*KtrainCAX(st(i));
ypred =Kp*((Kt+lamb*eye(Ntrain))\ytrain);
LtestCAX(i,ii)=rms(ytest-ypred);


end
end

%% delta
d=log(min(LtestAR,[],1)./min(LtestCA,[],1));
ds=log(min(LtestAR,[],1)./min(LtestCAS,[],1));
dx=log(min(LtestAR,[],1)./min(LtestCAX,[],1));

D(j)=nanmean(d);
Ds(j)=nanmean(ds);
Dx(j)=nanmean(dx);

Lar(j)=nanmean(min(LtestAR,[],1));
Lca(j)=nanmean(min(LtestCA,[],1));
Lcs(j)=nanmean(min(LtestCAS,[],1));
Lcx(j)=nanmean(min(LtestCAX,[],1));

disp(j*100/np)
end

%% histogram
% figure,
% [h,idx]=hist(D,10);
% bar(idx,h,'FaceColor',[1 0.5 0.5],'FaceAlpha',0.5);
% hold on;
% [h,idx]=hist(Ds,10);
% bar(idx,h,'FaceColor',[0.5 1 0.5],'FaceAlpha',0.5);
% hold on;
% [h,idx]=hist(Dx,10);
% bar(idx,h,'FaceColor',[0.5 0.5 1],'FaceAlpha',0.5);
% grid;

%% sigma
figure,
loglog(sx,nanmean(LtestAR,2),'b',sx,nanmean(LtestCA,2),'r',sx,...
    nanmean(LtestCAS,2),'g',sx,nanmean(LtestCAX,2),'k');grid

%%
figure,
plot(C,D,'r',C,Ds,'g',C,Dx,'k');grid;