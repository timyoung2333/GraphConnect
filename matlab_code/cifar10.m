clear;clc;
rootDir = './';
run([rootDir,'matconvnet_extended/matlab/vl_setupnn_extended']);
addpath([rootDir, 'utils/']);
load([rootDir, 'data/cifar10.mat']);

runWD = 1;
runGR = 0;

%% add layer-wise learning rate

%% ini. net
rng(0) ;

%%

f = 0.01;

% Define network CIFAR10-quick
net.layers = {} ;

% Block 1

M1=3;
M2=32;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,M1,M2, 'single'), zeros(1, M2, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

% w = repmat(reshape(psi5,5,5,1,K),[1,1,M1,M1]);
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{ single(w), zeros(1, K*M1, 'single')}}, ...
%                            'learningRate', [0.0,0.0], ...
%                            'stride', 1, ...
%                            'pad', 0) ;
%                        
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{ f*randn(1,1,K*M1,M2, 'single'), zeros(1, M2, 'single') }}, ...
%                            'stride', 1) ;
%                            
                                          
net.layers{end+1} = struct('type', 'relu') ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;


% Block 2


M1=32;
M2=64;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,M1,M2, 'single'), zeros(1,M2,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
                       
                       
% w = repmat(reshape(psi5,5,5,1,K),[1,1,M1,M1]);
% 
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{ single(w), zeros(1, K*M1, 'single')}}, ...
%                            'learningRate', [0.0,0.0], ...
%                            'stride', 1, ...
%                            'pad', 0) ;
%                        
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{ f*randn(1,1,K*M1,M2, 'single'), zeros(1, M2, 'single') }}, ...
%                            'stride', 1) ;                       
   
  
                         
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ; 

% Block 3, fc
M1 = 64;
M2 = 128;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,M1,M2, 'single'), zeros(1,M2,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type','relu');  

net.layers{end+1} = struct('type','dropout',...
                            'rate',0.5);


% M1=64;
% M2=96;
% w = repmat(reshape(psi5,5,5,1,K),[1,1,M1,M1]);
% 
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{ single(w), zeros(1, K*M1, 'single')}}, ...
%                            'learningRate', [0.0,0.0], ...
%                            'stride', 1, ...
%                            'pad', 2) ;
%                        
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{ f*randn(1,1,K*M1,M2, 'single'), zeros(1, M2, 'single') }}, ... 
%                            'stride', 1) ;   
%                        
%                        
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'avg', ...
%                            'pool', [3 3], ...
%                            'stride', 2, ...
%                            'pad', 0) ; % Emulate caffe

% Block 4, fc
M1 = 128;
M2 = 64;
net.layers{end+1} = struct('type','innerproduct',...
                            'weights',{{f*randn(1,1,M1,M2,'single'),zeros(1,M2,'single')}});
                        
net.layers{end+1} = struct('type','relu');     


net.layers{end+1} = struct('type','dropout',...
                            'rate',0.5);

% Block 5, fc
M1 = 64;
M2 = 10;
net.layers{end+1} = struct('type','innerproduct',...
                            'weights',{{f*randn(1,1,M1,M2,'single'),zeros(1,M2,'single')}});


% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;                

net0 = net;
clear net;


%%

% %% ini. net
% rng(0) ;
% 
% f=1/100 ;
% net0.layers = {} ;
% %1
% net0.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,3,96, 'single'), zeros(1, 96, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 2) ;
% net0.layers{end+1} = struct('type', 'relu') ;                       
% net0.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [3 3],...
%                            'stride', 2, ...
%                            'pad', 0) ;
% %2                      
% net0.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,96,128, 'single'),zeros(1,128,'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 2) ;
% net0.layers{end+1} = struct('type', 'relu') ;                       
% net0.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [3 3],...
%                            'stride', 2, ...
%                            'pad', 0) ;
% %full conn                       
% net0.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(4,4,128,256, 'single'),zeros(1,256,'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net0.layers{end+1} = struct('type', 'relu') ;                       
% net0.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [3 3], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
% 
% % full conn
% net0.layers{end+1} = struct('type','innerproduct',...
%                             'weights',{{f*randn(1,1,256,2048,'single'),zeros(1,2048,'single')}});
% net0.layers{end+1} = struct('type','relu');                        
% net0.layers{end+1} = struct('type','innerproduct',...
%                             'weights',{{f*randn(1,1,2048,2048,'single'),zeros(1,2048,'single')}});
% net0.layers{end+1} = struct('type','relu');                        
% 
% %net0.layers{end+1} = struct('type','dropout','rate',0.5);
%                        
% % softmax classfier
% net0.layers{end+1} = struct('type', 'innerproduct', ...
%                            'weights', {{f*randn(1,1,2048,10, 'single'),zeros(1,10,'single')}});
% net0.layers{end+1} = struct('type','softmaxloss');                       


numlayers =  numel(net0.layers) ;

%% vary training size
trId = 0;
paras.eta = 1e-3; 
paras.B = 100;
paras.E = 100;
paras.m = 0.9;
paras.GPU = 1;
paras.testlabel = labels(set==3);
paras.teststep = 1; 

for numTrain = [5000 ] %[50 100 400 700 1000 3000] 
    fprintf('training %d per class\n',numTrain);
    trId = trId+1;
    trainX = [];
    trainLabel = [];
    rng(0);
    
    % generate training and testing set
    for i=1:10        
        classi = find(labels==i&set==1);
        if numTrain<numel(classi)
            ids = randperm(numel(classi), numTrain);
            trainX = cat(4,trainX,single(data(:,:,:,classi(ids))));
            trainLabel = [trainLabel,i*ones(1,numTrain)];
        else
            trainX = cat(4,trainX,single(data(:,:,:,classi)));
            trainLabel = [trainLabel,i*ones(1,numel(classi))];
        end
    end   
    meanImg = mean(trainX,4);
    trainX = bsxfun(@minus,trainX,meanImg);
    paras.testdat = bsxfun(@minus,single(data(:,:,:,set==3)),meanImg);
	
   if runGR
      regId = 0;
      paras.w = 0;
      G=sparse(numTrain*10);
      for i=1:10
        dists_c = pdist(reshape(trainX(:,:,:,trainLabel==i),[],sum(trainLabel==i))');
        dists_c = exp(-(dists_c-mean(dists_c)).^2/var(dists_c));
        G(trainLabel==i,trainLabel==i)=squareform(dists_c);
      end
%       dists = pdist(reshape(trainX,[],size(trainX,4))');
%       dists = exp(-(dists-mean(dists)).^2/var(dists));
%       G = squareform(dists);
      paras.G = G;
      lam_range = [1e-6]; % [1e-6 5e-5 1e-5 5e-5 1e-4 5e-4];
      for lam = lam_range%[1e-5 5e-5 1e-4 5e-4 1e-3 5e-3]%[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2]
          regId = regId+1;
          paras.lambda = [zeros(1,numlayers-3),lam,0,0];
          [net{regId},info{regId}] = softmaxTrain_geoReg(trainX,trainLabel,net0,paras);
          acc(regId) = info{regId}.testAcc(end);
          trainLoss(regId) = info{regId}.trainLoss(end);
          testLoss(regId) = info{regId}.testLoss(end);
          generalization(regId) = info{regId}.testLoss(end)-info{regId}.trainLoss(end);
      end
      save(['gr_',num2str(numTrain),'.mat'],'net','info','acc','trainLoss','testLoss','generalization','lam_range');
    
   end

   if runWD
    clear net info acc trainLoss testLoss generalization;
    regId = 0;
    w_range = [1e-4]; %[1e-3 5e-3 1e-2 5e-2 1e-1] 
    for w = w_range %[1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1]
      regId = regId+1;
      paras.w = w;
      [net{regId},info{regId}] = softmaxTrain(trainX,trainLabel,net0,paras);
      acc(regId) = info{regId}.testAcc(end);
      trainLoss(regId) = info{regId}.trainLoss(end);
      testLoss(regId) = info{regId}.testLoss(end);
      generalization(regId) = info{regId}.testLoss(end)-info{regId}.trainLoss(end);
    end 
    save(['wd_',num2str(numTrain),'.mat'],'net','info','acc','trainLoss','testLoss','generalization','w_range'); 
    
   end

end
