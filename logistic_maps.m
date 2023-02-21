clc; close all; clear;

%% parameters
np=20;
nc=12;
ns=100;
p=2;
sx=logspace(-10,6,ns)';
lamb=1E-10;
N=2000;
Ntran=1500;

%% outputs
delta_linear   = zeros(np,1);
delta_stacked  = zeros(np,1);
delta_XK       = zeros(np,1);

surr_linear   = zeros(np,1);
surr_stacked  = zeros(np,1);
surr_XK       = zeros(np,1);

%% logistic model
C         = linspace(0,0.4,np)';
a         = 1.8;
s         = 0;
x         = zeros(N,1);
y         = zeros(N,1);
rx        = randn(N,1);
ry        = randn(N,1);

for i=1:np
c         = C(i);
N=2000;

for ii=2:N
    x(ii) = 1 - a*x(ii-1).^2 + s*rx(ii);
    y(ii) = (1-c)*(1-a*y(ii-1).^2) + c*(1-a*x(ii-1).^2) + s*ry(ii);
end

%% eliminate transient
x=x(Ntran+1:end);
y=y(Ntran+1:end);
N=N-Ntran;

%% main function
[delta_linear(i),delta_stacked(i),delta_XK(i),surr_linear(i),surr_stacked(i),surr_XK(i)]...
    =test_Granger_methods(y,x,p,nc,ns,sx,lamb);

disp(i*100/np);
end

%% plot results
figure,
plot(C,delta_linear,'k',C,delta_stacked,'b',C,delta_XK,'r','linewidth',2);grid;
hold on;
plot(C,surr_linear,'--k',C,surr_stacked,'--b',C,surr_XK,'--r','linewidth',2);grid;