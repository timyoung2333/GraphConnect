function [res1, res2] = vl_DRTfb(net, x1, x2, targetValue)


res1 = vl_simplenn_extended(net,x1);
res2 = vl_simplenn_extended(net,x2);

deltas = squeeze(res1(end).x) - squeeze(res2(end).x);
rhos = sum(deltas.^2 , 1);

dzdy1 = permute(bsxfun(@times,targetValue.*exp(-rhos),deltas), [3 4 1 2]);
dzdy2 = -dzdy1;

res1 = vl_simplenn_extended(net, x1, dzdy1, res1, 'onlyGradient', true);
res2 = vl_simplenn_extended(net, x2, dzdy2, res2, 'onlyGradient', true);

