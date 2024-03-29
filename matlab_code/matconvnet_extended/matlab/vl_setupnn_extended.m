function vl_setupnn_extended
% VL_SETUPNN  Setup the MatConvNet toolbox
%    VL_SETUPNN() function adds the MatConvNet toolbox to MATLAB path.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

root = vl_rootnn() ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', 'simplenn')) ;
%addpath(fullfile(root, 'matlab', 'dagnn')) ;
addpath(fullfile(root, 'matlab', 'xtest')) ;
addpath(fullfile(root, 'matlab', 'extensions0.1')) ;