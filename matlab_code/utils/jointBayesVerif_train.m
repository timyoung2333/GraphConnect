function [dataMean, PA, PG] = jointBayesVerif_train(Xtrain,classLabel,doEM)
% Implement the training stage of paper "Bayesian Face Revisited: A joint Formulation"
% Each image x is modeled as x=u+e,
% where u, e are independent. And p(u)=N(u;0,S_u), p(e)=N(e;0,S_e)
% u captures the class means' distribution. e captures the variation within a class
% First use EM to learn S_u and S_e on the training set. 
% Then for a testing pair (x_1,x_2), compute likelihood ratio p(x_1,x_2|same class)/p(x_1,x_2|different classes)
%
% Input-
% Xtrain: each column is a training data sample
% classLabel: class label of each training data sample
% doEM: if 1 (default), do EM iterates to refine the initializatin of Su and Se
%
% Output-
% PA, PG: projector associated with Su and Se

if nargin==2
    doEM = 1;
end

%% training stage
rng(0);
[n,N] = size(Xtrain);
K = max(classLabel);
dataMean = mean(Xtrain,2);
Xtrain = bsxfun(@minus,Xtrain,dataMean);

% initialize Su and Se as inter-class and intra-class variance (akin to LDA)
Se = zeros(n);
m = zeros(1,K);
for k=1:K % for each class
    m(k) = sum(classLabel==k);
    sumk(:,k) = sum(Xtrain(:,classLabel==k),2);
    % compute within-class covariance for each class
    cov_k = cov(Xtrain(:,classLabel==k)');
    Se = Se+cov_k*(m(k)-1)/(N-1);
end
Su = cov(Xtrain')-Se;

% if indicated, do EM iterates for better estimates of Su and Se
if doEM
    u = zeros(n,K);
    e = zeros(n,N);
    
    for iter = 1:5
        SuSe = Su/Se;
        for k=1:K
            u(:,k) = (SuSe*m(k)+eye(n))\(SuSe*sumk(:,k));
            e(:,classLabel==k) = bsxfun(@minus, Xtrain(:,classLabel==k), u(:,k));
        end
        Su = cov(u');
        Se = cov(e');
    end
    
end

inv_SuSe = inv(Su+Se);
A = inv_SuSe-inv((Su+Se)-Su*inv_SuSe*Su);
G = -inv(2*Su+Se)*Su*inv(Se);
[UA,eigA] = eig(-A);
[UG,eigG] = eig(-G);
sA = find(cumsum(diag(eigA))/sum(diag(eigA))>0.9,1);
sG = find(cumsum(diag(eigG))/sum(diag(eigG))>0.9,1);
s = max(sA,sG);
PA = UA(:,1:s)*sqrt(eigA(1:s,1:s));
PG = UG(:,1:s)*sqrt(eigG(1:s,1:s));





