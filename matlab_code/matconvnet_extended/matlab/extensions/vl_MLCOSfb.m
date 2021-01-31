function [res1, res2, objective] = vl_MLCOSfb(net, x1, x2, pLabels)
% compute pairwise loss,
% sum_i (net(x1_i)'*net(x2_i)-pLabels_i)^2
% Here the last layer of the net normalizes feature vector to mudular 1.
% plabels_i = 1 if x1_i and x2_i are from the same class, and -1 otherwise.

n = numel(net.layers) ;
opts.sync = true ;
gpuMode = isa(x1, 'gpuArray') ;

if size(pLabels,1)>1
    pLabels = pLabels';
end


%%%%% a forward pass for the two replicated structures %%%%%
% output of network 1
res1 = vl_simplenn_extended(net,x1);
clear x1;
% output of network 2
res2 = vl_simplenn_extended(net,x2);
clear x2;

%%%% compute gradients in the two replications %%%%
cos_ij = dot(squeeze(res1(end).x),squeeze(res2(end).x));
delta_ij = cos_ij-pLabels;
objective = sum(delta_ij.^2);

dzdy1 = bsxfun(@times,delta_ij,squeeze(res2(end).x));
dzdy1 = permute(dzdy1,[3 4 1 2]);
dzdy2 = bsxfun(@times,delta_ij,squeeze(res1(end).x));
dzdy2 = permute(dzdy2,[3 4 1 2]);

res1 = vl_simplenn_extended(net,res1(1).x,dzdy1,res1,'onlyGradient',true);
res2 = vl_simplenn_extended(net,res2(1).x,dzdy2,res2,'onlyGradient',true);

