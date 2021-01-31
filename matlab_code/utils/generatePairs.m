function pairs = generatePairs(labels,posRate,negRate)
% Generate pairs
% Input-
% labels: a vector of class label
% posRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all intra-class pairs. Otherwise, select 'posRate' of those intra-class pairs.
% negRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all inter-class pairs. Otherwise, select 'negRate' of those inter-class pairs.
%   If a char '=', the number of selected negative pairs equals that
%   of the positive pairs
% Ouput-
% pairs: a 3-column array, pairs(k,1), pairs(k,2) are the k-th
%   pair's indices. And pairs(k,3)=1 indicates the same class, 0 otherwise.




if ~exist('posRate','var') || isempty(posRate)
    posRate = 1;
end
if ~exist('negRate','var') || isempty(negRate)
    negRate = 1;
end

if posRate<=0||posRate>1
    error('rate of choosing a positive pair must be in (0,1]');
end

if ~isequal(negRate,'=') && (negRate>1||negRate<=0)
    error('rate of choosing a negative pair must be in (0,1], or assign =');
end



if size(labels,2)>1
    labels = labels';% make a column vector
end
K = max(labels);
N = length(labels);

if posRate==1 && negRate==1
    simm = single(bsxfun(@eq, labels, labels'));
    pairs = [nchoosek(1:N,2),simm(tril(true(N),-1))];
    return;
end

rng(0);
pairs = [];
for k=1:K
    class_k = find(labels==k);
    ps = nchoosek(1:numel(class_k),2);
    inds = [class_k(ps(:,1)) class_k(ps(:,2))];
    if posRate<1
        inds = inds(randperm(size(inds,1),ceil(posRate*size(inds,1))),:);
    end
    pairs = [pairs; [inds, ones(size(inds,1),1)] ];
    class_notk = setdiff(1:N,class_k);
    [a,b] = meshgrid(class_k,class_notk);
    inds_neg = [a(:), b(:)];
    if strcmp(negRate,'=')
        inds_neg = inds_neg(randperm(size(inds_neg,1), size(inds,1)),:);
    else
        inds_neg = inds_neg(randperm(size(inds_neg,1), round(size(inds_neg,1)*negRate)),:);
    end
    pairs = [pairs; [inds_neg,zeros(size(inds_neg,1),1)] ];
end



