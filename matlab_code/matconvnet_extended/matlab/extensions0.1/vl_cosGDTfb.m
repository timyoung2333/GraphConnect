function [res1, res2, objective_thisBatch] = vl_cosGDTfb(net, x1, x2, targetValue)


res1 = vl_simplenn_extended(net,x1);
res2 = vl_simplenn_extended(net,x2);

deltas = dot(squeeze(res1(end).x) , squeeze(res2(end).x)) - targetValue;
objective_thisBatch = mean(deltas.^2);

dzdy1 = permute(bsxfun(@times,deltas,squeeze(res2(end).x)), [3 4 1 2]);
dzdy2 = permute(bsxfun(@times,deltas,squeeze(res1(end).x)), [3 4 1 2]);

res1 = vl_simplenn_extended(net, x1, dzdy1, res1, 'onlyGradient', true);
res2 = vl_simplenn_extended(net, x2, dzdy2, res2, 'onlyGradient', true);

