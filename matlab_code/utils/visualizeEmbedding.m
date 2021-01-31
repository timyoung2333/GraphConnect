function ebd=visualizeEmbedding(x,dim,varargin)
% visualize embedding via pca or Laplacian map
%
% mandatory inputs-
% x: to-be-embedded data, each column is a data point
% dim: embedding dimension, must be 2 or 3
%
% optional inputs-
% method: 'rand', 'pca' or 'lmap' (Lapacian Map, default)
% labels: class label, if not provided, considered as a single class
% ordered: preserve the order of data pts within a class. It matters if we want to
%   see the effectiveness of a structure-preserving embedding algorithm.
%   default 1.
% kn: number of nearest neighors when constructing graph for embedding, default 10
% sigma: bandwidth parameter when constructing graph for embedding, default inf

ip = inputParser;
addRequired(ip,'x');
addRequired(ip,'dim',@(dim) (dim==1||dim==2||dim==3)&&(dim<=size(x,1)));
addOptional(ip,'method','lmap', @(method) strcmp(method,'lmap')||strcmp(method,'pca')||strcmp(method,'rand')||(isnumeric(method)&&size(method,1)==dim&&size(method,2)==size(x,1)));
addOptional(ip,'labels',ones(1,size(x,2)), @(labels) length(labels)==size(x,2));
addOptional(ip,'ordered',1, @(ordered) ordered==1 || ordered==0);
addOptional(ip,'kn',10, @isscalar);
addOptional(ip,'sigma',inf, @isscalar);

parse(ip,x,dim,varargin{:});
[n,N] = size(x);
numClass = max(ip.Results.labels);
markerPool = {'o','+','.','*','d','s','x','v','>','<','p','h'};

if n>dim
    if strcmp(ip.Results.method,'pca')
        ebd = pca2(x',dim);
        ebd = ebd';
    elseif strcmp(ip.Results.method,'lmap')
        ebd = Lmap(x,dim, ip.Results.kn, ip.Results.sigma);
    elseif strcmp(ip.Results.method,'rand')
        rng(0);
        ebd = randn(dim,n)/sqrt(dim)*x;
    else 
        ebd = ip.Results.method*x;
    end
else
    ebd = x;
end

if dim==1
    if ip.Results.ordered==1
        for c=1:numClass
            scatter(ebd(ip.Results.labels==c),zeros(1,sum(ip.Results.labels==c)),50,1:sum(ip.Results.labels==c),'marker',markerPool{c},'linewidth',2);
            hold on;
        end
    else
        scatter(ebd,zeros(1,N),50,ip.Results.labels,'marker','o','linewidth',2);
    end
elseif dim==2
    if ip.Results.ordered==1
        for c=1:numClass
            scatter(ebd(1,ip.Results.labels==c),ebd(2,ip.Results.labels==c),50,1:sum(ip.Results.labels==c),'marker',markerPool{c},'linewidth',2);
            hold on;
        end
    else
        scatter(ebd(1,:),ebd(2,:),50,ip.Results.labels,'marker','o','linewidth',2);
    end
elseif dim==3
    if ip.Results.ordered==1
        for c=1:numClass
            scatter3(ebd(1,ip.Results.labels==c),ebd(2,ip.Results.labels==c),ebd(3,ip.Results.labels==c),50,1:sum(ip.Results.labels==c),'marker',markerPool{c},'linewidth',2);
            hold on;
        end
    else
        scatter3(ebd(1,:),ebd(2,:),ebd(3,:),50,ip.Results.labels,'marker','o','linewidth',2);
    end
    
end

