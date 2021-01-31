function [y,dzd_weights,dzd_biase] = vl_nninnerproduct(x, filter, biase, dzdy)
% VL_NNRELU  CNN rectified linear unit
%   y = vl_nninnerproduct(x, filter, biase) applies the fully connected layer to the data
%   x. x can have arbitrary size.
%
%   [dzdx,dzd_weights,dzd_biase] = vl_nninnerproduct(x, filter, biase, dzdy) 
%   computes the network derivative DZDX with respect to the input X given the 
%   derivative DZDY with respect to the output Y. DZDX has the same dimension as X.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

cubeSiz = size(x);
x = reshape(x,1,1,[],cubeSiz(4));
if nargin==3
    y = vl_nnconv(x,filter,biase, 'stride',1,'pad',0);
elseif nargin==4
    [y,dzd_weights,dzd_biase] = vl_nnconv(x,filter,biase,dzdy, 'stride',1,'pad',0); % y is the derivative of network output w.r.t. input
    y = reshape(y,cubeSiz);
end
