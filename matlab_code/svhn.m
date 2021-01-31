clear;clc;
rootDir = './';
run([rootDir,'matconvnet_extended/matlab/vl_setupnn_extended']);
addpath([rootDir, 'utils/']);
load([rootDir,'data/svhn.mat'],'extrain_data','extrain_label','test_data','test_label');

runWD = 0;
runGR = 1;

%% ini. net
rng(0) ;

f=1/100 ;
net0.layers = {} ;
%1
net0.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,3,96, 'single'), zeros(1, 96, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 2) ;
net0.layers{end+1} = struct('type', 'relu') ;                       
net0.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3],...
                           'stride', 2, ...
                           'pad', 0) ;
%2                      
net0.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,96,128, 'single'),zeros(1,128,'single')}}, ...
                           'stride', 1, ...
                           'pad', 2) ;
net0.layers{end+1} = struct('type', 'relu') ;                       
net0.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3],...
                           'stride', 2, ...
                           'pad', 0) ;
%3                       
net0.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,128,256, 'single'),zeros(1,256,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net0.layers{end+1} = struct('type', 'relu') ;                       
net0.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% full conn
net0.layers{end+1} = struct('type','innerproduct',...
                            'weights',{{f*randn(1,1,256,2048,'single'),zeros(1,2048,'single')}});
net0.layers{end+1} = struct('type','relu');                        
net0.layers{end+1} = struct('type','innerproduct',...
                            'weights',{{f*randn(1,1,2048,2048,'single'),zeros(1,2048,'single')}});
net0.layers{end+1} = struct('type','relu');                        

%net0.layers{end+1} = struct('type','dropout','rate',0.5);
                       
% softmax classfier
net0.layers{end+1} = struct('type', 'innerproduct', ...
                           'weights', {{f*randn(1,1,2048,10, 'single'),zeros(1,10,'single')}});
net0.layers{end+1} = struct('type','softmaxloss');                       

%% vary training size
trId = 0;
paras.eta = 5e-3;
paras.B = 100;
paras.E = 100;
paras.m = 0.9;
paras.GPU = 1;
paras.testlabel = test_label;
paras.teststep = 1; 

for numTrain = [100 400 700 5000] 
    fprintf('training %d per class\n',numTrain);
    trId = trId+1;
    trainX = [];
    trainLabel = [];
    rng(0);
    % generate training and testing set
    for i=1:10        
        classi = find(extrain_label==i);
        if numTrain<numel(classi)
            ids = randperm(numel(classi), numTrain);
            trainX = cat(4,trainX,single(extrain_data(:,:,:,classi(ids))));
            trainLabel = [trainLabel,i*ones(1,numTrain)];
        else
            trainX = cat(4,trainX,single(extrain_data(:,:,:,classi)));
            trainLabel = [trainLabel,i*ones(1,numel(classi))];
        end
    end   
    meanImg = mean(trainX,4);
    trainX = bsxfun(@minus,trainX,meanImg);
    paras.testdat = bsxfun(@minus,single(test_data),meanImg);
	
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
      lam_range =  [5e-3 1e-2 5e-2]%[1e-6 5e-5 1e-5 5e-5 1e-4 5e-4 1e-3];
      for lam = lam_range%[1e-5 5e-5 1e-4 5e-4 1e-3 5e-3]%[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2]
          regId = regId+1;
          paras.lambda = [zeros(1,12),lam,0,0];
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
    w_range = [0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3];
    for w= w_range%[1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1]
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
