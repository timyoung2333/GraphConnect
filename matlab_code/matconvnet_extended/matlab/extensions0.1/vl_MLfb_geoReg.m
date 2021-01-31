function [res1, res2, objective, dtau] = vl_MLfb_geoReg(net, x1, x2, lambda, tau, pLabels, g)
% compute pairwise loss,
% metric learning+graph regularization
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
objective = sum(max(pairDist(pLabels==1)-tau,0)) + sum(max(tau-pairDist(pLabels==-1),0));
dtau = gather(sum(pLabels==-1 & pairDist<tau)-sum(pLabels==1 & pairDist>tau));

n = numel(net.layers) ;
dzdy = bsxfun(@times,delta_ij,pLabels+single(pLabels==-1 & pairDist>tau)-single(pLabels==1 & pairDist<tau))+...
        lambda(n)*bsxfun(@times, g, delta_ij);
dzdy = permute(dzdy,[3 4 1 2]);
res1(n+1).dzdx = dzdy;
res2(n+1).dzdx = -dzdy;
g = permute(g,[2 3 4 1]);

for i=n:-1:1

    l = net.layers{i} ;
    switch l.type
      case 'conv'
        [res1(i).dzdx, res1(i).dzdw{1}, res1(i).dzdw{2}] = ...
                vl_nnconv(res1(i).x, l.weights{1}, l.weights{2}, ...
                          res1(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride) ;
	    [res2(i).dzdx, res2(i).dzdw{1}, res2(i).dzdw{2}] = ...
                vl_nnconv(res2(i).x, l.weights{1}, l.weights{2}, ...
                          res2(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride) ;

      case 'convt'
        [res1(i).dzdx, res1(i).dzdw{1}, res1(i).dzdw{2}] = ...
                vl_nnconvt(res1(i).x, l.weights{1}, l.weights{2}, ...
                          res1(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample) ;
	    [res2(i).dzdx, res2(i).dzdw{1}, res2(i).dzdw{2}] = ...
                vl_nnconvt(res2(i).x, l.weights{1}, l.weights{2}, ...
                          res2(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample) ;

      case 'innerproduct' % added
        [res1(i).dzdx, res1(i).dzdw{1}, res1(i).dzdw{2}] = ...
            vl_nninnerproduct(res1(i).x, l.weights{1}, l.weights{2}, ...
                      res1(i+1).dzdx);
	    [res2(i).dzdx, res2(i).dzdw{1}, res2(i).dzdw{2}] = ...
            vl_nninnerproduct(res2(i).x, l.weights{1}, l.weights{2}, ...
                      res2(i+1).dzdx);

      case 'pool'
        res1(i).dzdx = vl_nnpool(res1(i).x, l.pool, res1(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method) ;
        res2(i).dzdx = vl_nnpool(res2(i).x, l.pool, res2(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method) ;
      case 'normalize'
        res1(i).dzdx = vl_nnnormalize(res1(i).x, l.param, res1(i+1).dzdx) ;
	    res2(i).dzdx = vl_nnnormalize(res2(i).x, l.param, res2(i+1).dzdx) ;
      case 'channelnormalize' %added
        res1(i).dzdx = vl_nnchannelnormalize(res1(i).x,res1(i+1).dzdx);
        res2(i).dzdx = vl_nnchannelnormalize(res2(i).x,res2(i+1).dzdx);
      case 'softmax'
        res1(i).dzdx = vl_nnsoftmax(res1(i).x, res1(i+1).dzdx) ;
        res2(i).dzdx = vl_nnsoftmax(res2(i).x, res2(i+1).dzdx) ;
      case 'loss'
        res1(i).dzdx = vl_nnloss(res1(i).x, l.class, res1(i+1).dzdx) ;
        res2(i).dzdx = vl_nnloss(res2(i).x, l.class, res2(i+1).dzdx) ;
      case 'softmaxloss'
        res1(i).dzdx = vl_nnsoftmaxloss(res1(i).x, l.class, res1(i+1).dzdx) ;
        res2(i).dzdx = vl_nnsoftmaxloss(res2(i).x, l.class, res2(i+1).dzdx) ;
      case 'relu'
        if ~isempty(res1(i).x)
          res1(i).dzdx = vl_nnrelu(res1(i).x, res1(i+1).dzdx) ;
        else
          res1(i).dzdx = vl_nnrelu(res1(i+1).x, res1(i+1).dzdx) ;
        end
        if ~isempty(res2(i).x)
          res2(i).dzdx = vl_nnrelu(res2(i).x, res2(i+1).dzdx) ;
        else
          res2(i).dzdx = vl_nnrelu(res2(i+1).x, res2(i+1).dzdx) ;
        end
      case 'sigmoid'
        res1(i).dzdx = vl_nnsigmoid(res1(i).x, res1(i+1).dzdx) ;
        res2(i).dzdx = vl_nnsigmoid(res2(i).x, res2(i+1).dzdx) ;
      case 'tanh'
        res1(i).dzdx = vl_nntanh(res1(i).x,res1(i+1).dzdx); 
        res2(i).dzdx = vl_nntanh(res2(i).x,res2(i+1).dzdx); 
      case 'noffset'
        res1(i).dzdx = vl_nnnoffset(res1(i).x, l.param, res1(i+1).dzdx) ;
        res2(i).dzdx = vl_nnnoffset(res2(i).x, l.param, res2(i+1).dzdx) ;
      case 'spnorm'
        res1(i).dzdx = vl_nnspnorm(res1(i).x, l.param, res1(i+1).dzdx) ;
        res2(i).dzdx = vl_nnspnorm(res2(i).x, l.param, res2(i+1).dzdx) ;
      case 'dropout'
        if opts.disableDropout
          res1(i).dzdx = res1(i+1).dzdx ;
          res2(i).dzdx = res2(i+1).dzdx ;
        else
          res1(i).dzdx = vl_nndropout(res1(i).x, res1(i+1).dzdx, ...
                                     'mask', res1(i+1).aux) ;
          res2(i).dzdx = vl_nndropout(res2(i).x, res2(i+1).dzdx, ...
                                     'mask', res2(i+1).aux) ;
        end
      case 'bnorm'
        [res1(i).dzdx, res1(i).dzdw{1}, res1(i).dzdw{2}] = ...
                vl_nnbnorm(res1(i).x, l.weights{1}, l.weights{2}, ...
                           res1(i+1).dzdx) ;
        [res2(i).dzdx, res2(i).dzdw{1}, res2(i).dzdw{2}] = ...
                vl_nnbnorm(res2(i).x, l.weights{1}, l.weights{2}, ...
                           res2(i+1).dzdx) ;
      case 'pdist'
        res1(i).dzdx = vl_nnpdist(res1(i).x, l.p, res1(i+1).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
        res2(i).dzdx = vl_nnpdist(res2(i).x, l.p, res2(i+1).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
      case 'custom'
        res1(i) = l.backward(l, res1(i), res1(i+1)) ;
        res2(i) = l.backward(l, res2(i), res2(i+1)) ;
    end

    res1(i+1).dzdx = [];
    res2(i+1).dzdx = [];

  end
  
  if i>1&&lambda(i-1)
	delta_ij = res1(i).x-res2(i).x;
	res1(i).dzdx = res1(i).dzdx+lambda(i-1)*bsxfun(@times, g, delta_ij);
	res2(i).dzdx = res2(i).dzdx-lambda(i-1)*bsxfun(@times, g, delta_ij);
  end


end



