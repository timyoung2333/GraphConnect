function pairs = generatePairs3(labels,posRate,negRate,lambda,Xtrain)
% Generate pairs
% Input-
% labels: a vector of class label
% posRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all intra-class pairs. Otherwise, select 'posRate' of those intra-class pairs.
% negRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all inter-class pairs. Otherwise, select 'negRate' of those inter-class pairs.
%   If a char '=', the number of selected negative pairs equals that
%   of the positive pairs
% lambda: a trade-off parameter for balancing between structure preserving
%   and label matching. The target value will be 
%   (lambda*L_ij + (1-lambda)*CS_ij) /2, default 1.
%   where L_ij=1 for positive pair and -1 for negative pair.
% Xtrain: each column is a training sample, will be used for computing CS_ij, must be given if lambda~=1
% Ouput-
% pairs: a 3-column array, pairs(k,1), pairs(k,2) are the k-th
%   pair's indices.

% asign default value
if ~exist('posRate','var') || isempty(posRate)
    posRate = 1;
end
if ~exist('negRate','var') || isempty(negRate)
    negRate = 1;
end
if ~exist('lambda','var') || isempty(lambda)
    lambda = 1;
end

% sanity check
if posRate<=0||posRate>1
    error('rate of choosing a positive pair must be in (0,1]');
end

if ~isequal(negRate,'=') && (negRate>1||negRate<=0)
    error('rate of choosing a negative pair must be in (0,1], or assign =');
end

if lambda>1 || lambda<0 
    error('lambda must be in [0,1]');
end

if lambda~=1 && (~exist('Xtrain','var') || isempty(Xtrain))
    error('must give training pairs')
end

% normalize data
if lambda~=1
    Xtrain = normcol(Xtrain);
end

% generate pairs
if size(labels,1)>1
    labels = labels'; % make it row vector
end
N = length(labels);

if posRate==1 && negRate==1
    simm = single(bsxfun(@eq, labels, labels'));
    simm(simm==0) = -1;
    if lambda~=1
        CS_ij = Xtrain'*Xtrain;
        simm = (lambda*simm+(1-lambda)*CS_ij)/2;
    end
    pairs = nchoosek(1:length(labels),2);
    pairs = [pairs, simm( sub2ind(size(simm),pairs(:,1),pairs(:,2)) )];
else
    rng(0);
    pairs = [];
    for k=unique(labels)
        class_k = find(labels==k)';
        ps = nchoosek(1:numel(class_k),2);       
        if posRate<1
            subInd = randperm(size(ps,1),ceil(posRate*size(ps,1)));
            ps = ps(subInd,:);
        end
        inds = [class_k(ps(:,1)) class_k(ps(:,2))];
       
        if lambda~=1
            CS_ij = dot(Xtrain(:,inds(:,1)),Xtrain(:,inds(:,2)));
            pairs = [pairs; [inds, 0.5*(ones(size(inds,1),1)*lambda+CS_ij'*(1-lambda)) ]];
        else
            pairs = [pairs; [inds, ones(size(inds,1),1)] ];
        end
        
        % put in appropriate number of negative pairs
        class_notk = setdiff(1:N,class_k);
        [a,b] = meshgrid(class_k,class_notk);
        inds_neg = [a(:), b(:)];
        if strcmp(negRate,'=')
            inds_neg = inds_neg(randperm(size(inds_neg,1), size(inds,1)),:);
        else
            inds_neg = inds_neg(randperm(size(inds_neg,1), round(size(inds_neg,1)*negRate)),:);
        end
        if lambda~=1
            CS_ij = dot(Xtrain(:,inds_neg(:,1)), Xtrain(:,inds_neg(:,2)));
            pairs = [pairs; [inds_neg, 0.5*(-ones(size(inds_neg,1),1)*lambda+CS_ij'*(1-lambda)) ] ];
        else
            pairs = [pairs; [inds_neg, -ones(size(inds_neg,1),1) ] ];
        end
    end
end



