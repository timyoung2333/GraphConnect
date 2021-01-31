function [simScore, pd, pf, accu, AUC] = simpleVerif(feature1,feature2,labels,sim)
% Compute a similarity score between two features and plot ROC curve
% Input-
% feature1(:,i) and feature2(:,i) constitute the i-th pair whose label is to-be
% determined. feature1 and feature2 must be of the same size.
% labels: vector whose length is size(feature1,2), positive label is 1.
%   negative label can be denoted as any number not 1.
% sim: similarity measure: can be 'cos', 'negEuc', 'cosAbs'
% Ouput-
% simScore: similarity score
% pd, pf: for ROC plot
% accu: the accuracy obtained via 10-fold cross-validation
% AUC: area under the ROC plot


if ~isequal(size(feature1),size(feature2))
    error('feature1 and feature2 must be of the same size');
end

if size(feature1,2)~=length(labels)
    error('length of label must be same as the number of columns in feature1 or feature2');
end

if size(labels,1)>1
    labels = labels'; % make it row vector
end

switch sim
    case 'cos'
        simScore = dot(feature1,feature2)./(sqrt(dot(feature1,feature1).*dot(feature2,feature2))); 
    case 'cosAbs'
        simScore = abs(dot(feature1,feature2)./(sqrt(dot(feature1,feature1).*dot(feature2,feature2)))); 
    case 'negEuc'
        simScore = -sum((feature1-feature2).^2,1);
end

% ROC curve
[pd,pf,~,AUC]=perfcurve(labels,simScore,1);


%%
% 10-fold cross validate to report accuracy, assuming balanced positive and negative samples
nFolds = 10;
foldSize_pos = round(sum(labels==1)/nFolds);
foldSize_neg = round(sum(labels~=1)/nFolds);
% permute positive and negative pairs
positiveInd = find(labels==1);
negativeInd = find(labels~=1);
rng(0);
positiveInd = positiveInd(randperm(length(positiveInd)));
negativeInd = negativeInd(randperm(length(negativeInd)));

for i=1:nFolds
    
    % hold out one test fold with balanced positive and negative samples
    test_ind_pos = positiveInd( (i-1)*foldSize_pos+1:min(i*foldSize_pos,length(positiveInd)) );
    test_ind_neg = negativeInd( (i-1)*foldSize_neg+1:min(i*foldSize_neg,length(negativeInd)) );
    test_ind = [test_ind_pos, test_ind_neg];
    
    % the rest 9 folds are for threshold tuning
    tune_ind = setdiff(1:length(labels),test_ind);
    
    % now find the optimal threshold on one fold of test samples
    [XX,YY,ths,~,OPTROCPT] = perfcurve(labels(tune_ind),simScore(tune_ind),1);
    optTh = ths((XX==OPTROCPT(1))&(YY==OPTROCPT(2)));
    
    % apply the optimal threshold to the test fold
    accu(i) = (sum(simScore(test_ind)>optTh & labels(test_ind)==1) + sum(simScore(test_ind)<optTh & labels(test_ind)~=1))/length(test_ind);
    
end
    
accu = mean(accu);
