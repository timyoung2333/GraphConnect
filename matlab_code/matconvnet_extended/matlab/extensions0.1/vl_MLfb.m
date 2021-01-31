function [res1, res2, objective, dtau] = vl_MLfb(net, x1, x2, tau, pLabels)
% compute pairwise loss,
% sum_i [-plabels_i*tau + plabels_i*|net(x1_i)-net(x2_i)|^2]_+ where []_+ is the hinge loss
% plabels_i = 1 if x1_i and x2_i are from the same class, and -1 otherwise.

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
delta_ij = squeeze(res1(end).x-res2(end).x);
pairDist = sum(delta_ij.^2,1);
objective = sum(max(pairDist(pLabels==1)-tau,0))+sum(max(tau-pairDist(pLabels==-1),0));


dzdy = bsxfun(@times,delta_ij,pLabels+single(pLabels==-1 & pairDist>tau)-single(pLabels==1 & pairDist<tau));
dzdy = permute(dzdy,[3 4 1 2]);
dtau = gather(sum(pLabels==-1 & pairDist<tau)-sum(pLabels==1 & pairDist>tau));

res1 = vl_simplenn_extended(net,res1(1).x,dzdy,res1,'onlyGradient',true);
res2 = vl_simplenn_extended(net,res2(1).x,-dzdy,res2,'onlyGradient',true);

