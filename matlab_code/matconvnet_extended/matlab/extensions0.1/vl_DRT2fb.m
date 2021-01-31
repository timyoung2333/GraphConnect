function [res1, res2, dtau] = vl_DRT2fb(net, x1, x2, tau, pLabels, r, lambda)


res1 = vl_simplenn_extended(net,x1);
res2 = vl_simplenn_extended(net,x2);


%%%% compute gradients in the two replications %%%%
delta_ij = squeeze(res1(end).x-res2(end).x);
pairDist = sum(delta_ij.^2,1);
neighbors = (r~=0);
r = full(r(neighbors));

dzdy = bsxfun(@times,delta_ij,pLabels+single(pLabels==-1 & pairDist>tau)-single(pLabels==1 & pairDist<tau));
dzdy(:,neighbors) = dzdy(:,neighbors)+lambda*bsxfun(@times,sign(pairDist(neighbors)-r),2*delta_ij(:,neighbors));
%dzdy(:,neighbors) = dzdy(:,neighbors)-lambda*bsxfun(@times,r,2*delta_ij(:,neighbors));

dzdy = permute(dzdy,[3 4 1 2]);
dtau = gather(sum(pLabels==-1 & pairDist<tau)-sum(pLabels==1 & pairDist>tau));

res1 = vl_simplenn_extended(net,res1(1).x,dzdy,res1,'onlyGradient',true);
res2 = vl_simplenn_extended(net,res2(1).x,-dzdy,res2,'onlyGradient',true);


