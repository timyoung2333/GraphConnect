function [simScore, pd, pf, ACC, AUC] = similarityVerification(feature,labels,sim)
% Compute a similarity score between two features and plot ROC curve
% Input-
% feature can be a 2-element cell, where 
% feature{1}(:,i) and feature{2}(:,i) constitute the i-th pair whose label is to-be
% determined. feature{1} and feature{2} must be of the same size. 
% feature can also be a matrix, where feature(:,i) is the i-th datum. 
% In this case all pairwise label will be inferred.
%
% labels: if feature is a 2-element cell, label(i) is 1/-1
% indicating if feature{1}(:,i) and feature{2}(:,i) is from the same class.
% If feature is a matrix, label(i) is the class label of feature(:,i)
%
% sim: similarity measure: can be 'cos', 'negEuc', etc
% Ouput-
% simScore: similarity score
% pd, pf: for ROC plot
% ACC: the accuracy obtained via 10-fold cross-validation
% AUC: area under the ROC plot


if size(labels,1)>1
    labels = labels'; % make it row vector
end
    
if iscell(feature)
    if ~isequal(size(feature{1}),size(feature{2})) || length(labels)~=size(feature{1},2)
        error('input dimension mismatch');
    end
    
    switch sim
        case 'cos'
            simScore = dot(normcol(feature{1}),normcol(feature{2}));
        case 'negEuc'
            simScore = -sum((feature{1}-feature{2}).^2,1);
    end
else
    if size(feature,2)~=length(labels)
        error('input dimension mismatch');
    end
    aff=bsxfun(@eq,labels,labels');
    aff(1:size(aff,2)+1:end)=0;
    labels=single(squareform(aff,'tovector'));
    switch sim
        case 'cos'
            feature = normcol(feature);
            simScore = feature'*feature;
            simScore(1:size(feature,2)+1:end) = 0;
            simScore = squareform(simScore,'tovector');
        case 'negEuc'
            simScore = -pdist(feature');
    end
    
end

% ROC curve
%[pd,pf,~,AUC]=perfcurve(labels,simScore,1);


%%
% 10-fold cross validate to report accuracy, splitting into balanced positive and negative samples
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
    ind_pos = positiveInd( (i-1)*foldSize_pos+1:min(i*foldSize_pos,length(positiveInd)) );
    ind_neg = negativeInd( (i-1)*foldSize_neg+1:min(i*foldSize_neg,length(negativeInd)) );
    ind = [ind_pos, ind_neg];
    labelCell{i} = labels(ind);
    scoreCell{i} = simScore(ind);    
end
[pf,pd,~,AUC,OPT] = perfcurve(labelCell,scoreCell,1);
ACC = (OPT(2)*length(positiveInd)+(1-OPT(1))*length(negativeInd))/length(labels);
