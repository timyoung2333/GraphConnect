function pairInfo = generatePairs5(labels,posRate,negRate,Xtrain)
% Generate pairs
% Input-
% labels: a vector of class label
% posRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all intra-class pairs. Otherwise, select 'posRate' of those intra-class pairs.
% negRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all inter-class pairs. Otherwise, select 'negRate' of those inter-class pairs.
%   If a char '=', the number of selected negative pairs equals that
%   of the positive pairs
% Xtrain: each column is a training sample, will be used for computing CS_ij
% Ouput-
% pairInfo: a 4-column array, pairs(k,1), pairs(k,2) are the k-th
%   pair's indices. pairs(k,3) is the label, 1/0. pairs(k,4) is the
%   inner product


% asign default value
if ~exist('posRate','var') || isempty(posRate)
    posRate = 1;
end
if ~exist('negRate','var') || isempty(negRate)
    negRate = 1;
end

% sanity check
if posRate<=0||posRate>1
    error('rate of choosing a positive pair must be in (0,1]');
end

if ~isequal(negRate,'=') && (negRate>1||negRate<0)
    error('rate of choosing a negative pair must be in [0,1], or assign =');
end

N = length(labels);

% generate pairs
if size(labels,1)>1
    labels = labels'; % make it row vector
end

if posRate==1 && negRate==1
    pairInfo = nchoosek(1:N,2);
    IDs = sub2ind([N N],pairInfo(:,1),pairInfo(:,2));
    
    simm = single(bsxfun(@eq, labels, labels'));
    CS = Xtrain'*Xtrain;
    
    pairInfo = [pairInfo, simm(IDs), CS(IDs)];
    
else
    rng(0);
    pairInfo = [];
    for k=unique(labels)
        class_k = find(labels==k)';
        ps = nchoosek(1:numel(class_k),2);       
        if posRate<1
            subInd = randperm(size(ps,1),ceil(posRate*size(ps,1)));
            ps = ps(subInd,:);
        end
        inds = [class_k(ps(:,1)) class_k(ps(:,2))];
        CS_ij = dot(Xtrain(:,inds(:,1)),Xtrain(:,inds(:,2)));
        pairInfo = [pairInfo; [inds, ones(size(inds,1),1), CS_ij']];
        
        % put in appropriate number of negative pairs
        if negRate~=0
            class_notk = setdiff(1:N,class_k);
            [a,b] = meshgrid(class_k,class_notk);
            inds_neg = [a(:), b(:)];
            if strcmp(negRate,'=')
                inds_neg = inds_neg(randperm(size(inds_neg,1), size(inds,1)),:);
            else
                inds_neg = inds_neg(randperm(size(inds_neg,1), ceil(size(inds_neg,1)*negRate)),:);
            end
            CS_ij = dot(Xtrain(:,inds_neg(:,1)), Xtrain(:,inds_neg(:,2)));
            pairInfo = [pairInfo; [inds_neg, zeros(size(inds_neg,1),1), CS_ij'] ];
        end
    end
end



