function pairInfo = gen_label_and_geometry_info(Xtrain,labels,k,selectNeg)
% Generate pairs
% Input-
% Xtrain: training data. If a 2D matrix, then each column is a feature
%   vector.
% labels: a vector of class label
% selectNeg: the ratio of selected negative pairs
% k: nearest k neighbors
% Output-
% pairInfo: a 4-column array, pairInfo(k,1), pairInfo(k,2) are the k-th
%   pair's indices. pairInfo(k,3) is the label, 1/0 (1 for same class). 
%   pairInfo(k,4) is the similarity

N = length(labels);
if size(labels,1)==1 % make sure 'labels' is a column vector 
    labels=labels';
end
Xtrain = reshape(Xtrain,[],N); % reshape Xtrain into column vectors if necessary

% the postive pairs will be weighted by their adjacency
pairInfo = [];
for c=1:max(labels)
    cInd = find(labels==c);
    [IDX, Dist] = knnsearch(Xtrain(:,cInd)',Xtrain(:,cInd)','K',k+1);
    IDX(:,1) = [];
    Dist(:,1) = []; % remove self-to-self
    
    % symmetrize
    adjacency = sparse(kron(ones(k,1),cInd), cInd(IDX(:)), true, N, N);
    adjacency = adjacency|adjacency';
    [i,j] = find(tril(adjacency,-1));
    
    pairInfo = [pairInfo;[i,j,ones(length(i),2)]];
end

%-- now for the negative pairs
if ~exist('selectNeg','var') % do not select negative pairs
    return;
end
% select some negative pairs 
rng(0);
pairIndicator = bsxfun(@eq, labels, labels');
pairIndicator(1:N+1:end) = false;
pairIndicator = squareform(pairIndicator);

negPairs = find(pairIndicator'==0);
if isnumeric(selectNeg)
    negPairs = negPairs(randperm(length(negPairs),ceil(length(negPairs)*selectNeg)));
elseif strcmp(selectNeg,'ba') % balanced positive and negative pairs
    negPairs = negPairs(randperm(length(negPairs),size(pairInfo,1)));
end

i = ceil(N-1/2-sqrt((N-1/2)^2-2*negPairs));
j = negPairs-(N-(i+1)/2).*i+N;
negPairs = [i,j];
pairInfo = [pairInfo;[negPairs,zeros(length(i),2)]];



