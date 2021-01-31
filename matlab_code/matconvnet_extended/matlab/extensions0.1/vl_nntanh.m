function y = vl_nntanh(x,dzdy)
% VL_NNRELU  CNN sigmoid linear unit
%   Y = VL_NNTANH(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNTANH(X, DZDY) computes the network derivative DZDX
%   with respect to the input X given the derivative DZDY with respect
%   to the output Y. DZDX has the same dimension as X.

% Copyright (C) 2015 Jiaji Huang.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

y = tanh(x);
if nargin <= 1 || isempty(dzdy)
   return;
else
  y = dzdy .* (1-y.^2) ;
end