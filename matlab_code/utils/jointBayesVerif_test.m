function [simScore, pd, pf, accu] = jointBayesVerif_test(feature1, feature2, labels, PA, PG, dataMean)
% testing stage of joint Bayesian verification: 
%
% compute ROC over the entire testing set

if size(labels,1)>1
    labels=labels'; % make labels a row vector
end
feature1 = bsxfun(@minus, feature1, dataMean);
feature2 = bsxfun(@minus, feature2, dataMean);


A1 = PA'*feature1;
A2 = PA'*feature2;
G1 = PG'*feature1;
G2 = PG'*feature2;

simScore = 2*dot(G1,G2)-dot(A1,A1)-dot(A2,A2);


rmin = min(simScore);
rmax = max(simScore);
pd = []; 
pf = [];
th_vals = linspace(rmin,rmax,100);
for th=th_vals
    pd = [pd sum(simScore>th & labels==1)/sum(labels==1)];
    pf = [pf sum(simScore>th & labels==0)/sum(labels==0)];
end


% 10-fold cross validation to pick the threshold
nFolds = 10;
foldSize = ceil(length(labels)/nFolds);
for i=1:nFolds
    % hold out the i-th fold as test, the others as validation for chooing threshold
    correctDecision = [];
    test_ind = (i-1)*foldSize+1:min(i*foldSize,length(labels));
    val_ind = setdiff(1:length(labels),test_ind);
    
    
    for th=th_vals
        correctDecision = [correctDecision sum(simScore(val_ind)>th & labels(val_ind)==1)+sum(simScore(val_ind)<th & labels(val_ind)==0)];
    end
    [~,kth] = max(correctDecision);
    th_star = th_vals(kth);
    
    accu(i) = (sum(simScore(test_ind)>th_star & labels(test_ind)==1) + sum(simScore(test_ind)<th_star & labels(test_ind)==0))/length(test_ind);
    
end
    
accu = mean(accu);