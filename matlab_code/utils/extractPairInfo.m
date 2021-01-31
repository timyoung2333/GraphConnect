function pairInfo = extractPairInfo(Xtrain,labels,posRate,negRate,simFunc)
% Generate pairs
% Input-
% labels: a vector of class label
% posRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all intra-class pairs. Otherwise, select 'posRate' of those intra-class pairs.
% negRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all inter-class pairs. Otherwise, select 'negRate' of those inter-class pairs.
%   If a char '=', the number of selected negative pairs equals that
%   of the positive pairs
% Xtrain: training data. If a 2D matrix, then each column is a feature
%   vector.
% simFunc: similarity function handle. Computing similarity between
% corresponding two columns in matrix X and Y. Examples are:
%   'cos' = @(X,Y) dot(bsxfun(@rdivide,X,sqrt(sum(X.^2,1))),bsxfun(@rdivide,Y,sqrt(sum(Y.^2,1))));
%   'euc' = @(X,Y) -sqrt(sum((X-Y).^2,1))
%
% Ouput-
% pairInfo: a 4-column array, pairInfo(k,1), pairInfo(k,2) are the k-th
%   pair's indices. pairInfo(k,3) is the label, 1/-1 (1 for same class). 
%   pairInfo(k,4) is the similarity


% sanity check
if posRate<=0||posRate>1
    error('rate of choosing a positive pair must be in (0,1]');
end

if ~isequal(negRate,'=') && (negRate>1||negRate<0)
    error('rate of choosing a negative pair must be in [0,1], or assign =');
end

% asign default value
if ~exist('posRate','var') || isempty(posRate)
    posRate = 1;
end
if ~exist('negRate','var') || isempty(negRate)
    negRate = 1;
end
if ~exist('simFunc','var') || isempty(simFunc)
    simFunc = @(X,Y) dot(bsxfun(@rdivide,X,sqrt(sum(X.^2,1))),bsxfun(@rdivide,Y,sqrt(sum(Y.^2,1))));
elseif strcmp(simFunc,'cos')
    simFunc = @(X,Y) dot(bsxfun(@rdivide,X,sqrt(sum(X.^2,1))),bsxfun(@rdivide,Y,sqrt(sum(Y.^2,1))));
elseif strcmp(simFunc,'euc')
    simFunc = @(X,Y) -sqrt(sum((X-Y).^2,1));
elseif ~isa(simFunc,'function_handle')
    error('5-th parameter must be a valid function handle or a already defined similarity (cos, euc)');
end


% generate label indicators for all the pairs
sizXtrain = size(Xtrain);
if length(sizXtrain)>2
    Xtrain = reshape(Xtrain,[],sizXtrain(end));
end
N = length(labels);
if N~=size(Xtrain,2)
    error('#labels must equal #training samples');
end
pairIndicator = bsxfun(@eq, labels, labels');
pairIndicator = pairIndicator(tril(true(N),-1));
posPairs = find(pairIndicator);
negPairs = find(pairIndicator==0);

% if not using all the pairs, randomly sample them
rng(0);
if posRate<1
    posPairs = posPairs(randperm(numel(posPairs),ceil(numel(posPairs)*posRate)));
end
if isnumeric(negRate) && negRate<1
    negPairs = negPairs(randperm(numel(negPairs),ceil(numel(negPairs)*posRate)));
elseif strcmp(negRate,'=')
    negPairs = negPairs(randperm(numel(negPairs),numel(posPairs)));
end


i = ceil(N-1/2-sqrt((N-1/2)^2-2*posPairs));
j = posPairs-(N-(i+1)/2).*i+N;
posPairs = [i,j];


i = ceil(N-1/2-sqrt((N-1/2)^2-2*negPairs));
j = negPairs-(N-(i+1)/2).*i+N;
negPairs = [i,j];

pairInfo = [posPairs,ones(size(posPairs,1),1),simFunc(Xtrain(:,posPairs(:,1)),Xtrain(:,posPairs(:,2)))';
            negPairs,-ones(size(negPairs,1),1),simFunc(Xtrain(:,negPairs(:,1)),Xtrain(:,negPairs(:,2)))'];
