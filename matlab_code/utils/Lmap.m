function [embedding,graph,eigVal] = Lmap(x,dim,kn,sigma)
% Laplacian map
%
% input-
% dim: embedding dimension
% kn: k nearest neighbor in neighborhood search
% sigma: sigma*maximum local distance = bandwidth in Gaussian kernel,
%       default is inf
%
% output-
% embedding: each column is a embeded data point
% graph: the constucted affinity matrix, which defines a graph
% eigVal: the smallest (dim) eigevalues, excluding 0

if nargin<3
    kn = 10;
end
if nargin<4
    sigma = inf;
end
N = size(x,2);
[idx,D] = knnsearch(x',x','K',kn);
sigma = sigma*max(D(:));
graph = exp(-(D/sigma).^2);
graph = sparse(kron(ones(1,kn),1:N),idx(:),graph(:),N,N);
graph = max(graph,graph'); % to symmetrize
D = sum(graph,2);L = diag(D)-graph; % lapacian matrix
D_halfInv = diag(D.^(-0.5));
[eigVec,eigVal] = eigs(D_halfInv*L*D_halfInv+1e-10*eye(N),dim+1,'SM');
embedding = eigVec(:,2:end)';
eigVal = diag(eigVal);