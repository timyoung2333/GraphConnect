function y = vl_nnchannelnormalize(x,dzdy)
% invoked when layer type is channelnormalize, normalize a feature vector
% to unit-modular.
% usage-
% y = vl_nnchannelnormalize(x)
% forward pass, compute normalized feature vector
% y = vl_nnchannelnormalize(x,dzdy)
% backward pass, compute the gradient of network output with respect to
% input x

norms = sqrt(sum(x.^2,3));
if nargin==1
    y = bsxfun(@rdivide,x,norms);
elseif nargin==2
    y = bsxfun(@rdivide,dzdy,norms)-bsxfun(@times,x,sum(bsxfun(@rdivide,dzdy.*x,norms.^3),3));
end
