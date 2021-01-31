function pairs = generatePairs2(labels,posRate,negRate,vals,pm,Xtrain)
% Generate pairs
% Input-
% labels: a vector of class label
% posRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all intra-class pairs. Otherwise, select 'posRate' of those intra-class pairs.
% negRate: optional parameter between 0 and 1, including 1 (default). 
%   If 1, use all inter-class pairs. Otherwise, select 'negRate' of those inter-class pairs.
%   If a char '=', the number of selected negative pairs equals that
%   of the positive pairs
% vals: choices of target value, 'bino'(default, 0 and 1)
%   'prod', values in intra-class gram matrix
%   'rank1', rank one approximation of intra-class gram matrix
% pm: if 1, use -1 to indicate inter-class. default 0
% Xtrain: training data of each class, must be given if vals='rank1' or 'prod'
% Ouput-
% pairs: a 3-column array, pairs(k,1), pairs(k,2) are the k-th
%   pair's indices. pairs(k,3)=0 indicate different classes
%  For intra class, pairs(k,3) is a value defined by the choice of 'vals'.



% asign default value
if ~exist('posRate','var') || isempty(posRate)
    posRate = 1;
end
if ~exist('negRate','var') || isempty(negRate)
    negRate = 1;
end
if ~exist('vals','var')|| isempty(vals)
    vals = 'bino';
end
if ~exist('pm', 'var')|| isempty(pm)
    pm = 0;
end

% sanity check
if posRate<=0||posRate>1
    error('rate of choosing a positive pair must be in (0,1]');
end

if ~isequal(negRate,'=') && (negRate>1||negRate<=0)
    error('rate of choosing a negative pair must be in (0,1], or assign =');
end
if (strcmp(vals,'rank1')||strcmp(vals,'prod')) && ~exist('Xtrain','var')
    error('must provide training data for estimate of target inner product');
end

% generate pairs
if size(labels,1)>1
    labels = labels'; % make it row vector
end
N = length(labels);

if posRate==1 && negRate==1
    simm = single(bsxfun(@eq, labels, labels'));
    if pm
        simm(simm==0) = -1;
    end
    switch vals
        case 'bino'
            pairs = [nchoosek(1:N,2),simm(tril(true(N),-1))];
        case 'prod'
            for c=unique(labels)
                class_c = (labels==c);
                gm=Xtrain(:,class_c)'*Xtrain(:,class_c);
                simm(class_c,class_c) = gm;
            end
            pairs = [nchoosek(1:N,2),simm(tril(true(N),-1))];
        case 'rank1'
            for c=unique(labels)
                class_c = (labels==c);
                gm=Xtrain(:,class_c)'*Xtrain(:,class_c);
                [u1,sig] = eig(gm);
                [sig,ord] = sort(abs(diag(sig)),'descend');
                gm = sig(1)*u1(:,ord(1))*u1(:,ord(1))';
                simm(class_c,class_c) = gm;
            end
            pairs = [nchoosek(1:N,2),simm(tril(true(N),-1))];
    end
        
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
        switch vals
            case 'bino'
                pairs = [pairs; [inds, ones(size(inds,1),1)] ];
            case 'prod'
                gm=Xtrain(:,class_k)'*Xtrain(:,class_k);
                pairs = [pairs; [inds, gm( sub2ind(size(gm),ps(:,1),ps(:,2)) )] ];
            case 'rank1'
                gm=Xtrain(:,class_k)'*Xtrain(:,class_k);
                [u1,sig] = eig(gm);
                [sig,ord] = sort(abs(diag(sig)),'descend');
                gm = sig(1)*u1(:,ord(1))*u1(:,ord(1))';
                pairs = [pairs; [inds, gm( sub2ind(size(gm),ps(:,1),ps(:,2)) )] ];
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
        if pm
            pairs = [pairs; [inds_neg,-ones(size(inds_neg,1),1)] ];
        else
            pairs = [pairs; [inds_neg,zeros(size(inds_neg,1),1)] ];
        end
    end
end



