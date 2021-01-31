clear;clc;
rootDir = './';
run([rootDir,'matconvnet_extended/matlab/vl_setupnn_extended']);
addpath([rootDir, 'utils/']);
load([rootDir,'data/mnist.mat']);


runWD = 1;
runGR = 1;
%% ini. net
rng(0) ;

f=1/100 ;
net0.layers = {} ;
net0.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net0.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net0.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,20,50, 'single'),zeros(1,50,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net0.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net0.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,50,500, 'single'),zeros(1,500,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net0.layers{end+1} = struct('type', 'relu') ;

% softmax classfier
net0.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,500,10, 'single'),zeros(1,10,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net0.layers{end+1} = struct('type','softmaxloss');                       

%% vary training size
trId = 0;
paras.eta = 1e-3;
paras.B = 100;
paras.E = 300;
paras.m = 0.9;
paras.GPU = 1;
paras.testlabel = labels(set==3);
paras.teststep = 1; 

for numTrain = [50 100:300:1000 3000 6000]
    fprintf('training %d per class\n',numTrain);
    trId = trId+1;
    trainX = [];
    trainLabel = [];
    rng(0);
    % generate training and testing set
    for i=1:10        
        classi = find(labels==i & set==1);
        if numTrain<numel(classi)
            ids = randperm(numel(classi), numTrain);
            trainX = cat(4,trainX,data(:,:,:,classi(ids)));
            trainLabel = [trainLabel,i*ones(1,numTrain)];
        else
            trainX = cat(4,trainX,data(:,:,:,classi));
            trainLabel = [trainLabel,i*ones(1,numel(classi))];
        end
    end   
    meanImg = mean(trainX,4);
    trainX = bsxfun(@minus,trainX,meanImg);
    paras.testdat = bsxfun(@minus,data(:,:,:,set==3),meanImg);

   if runGR
     regId = 0;
     paras.w = 0;
    for lam = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]
        regId = regId+1;
        paras.lambda = [0,0,0,0,0,lam,0,0];
        [net,info,G] = softmaxTrain_geoReg(trainX,trainLabel,net0,paras);
        if ~isfield(paras,'G')
            paras.G = G; % avoid recompute the same G
        end
        geoReg_acc(trId,regId) = info.testAcc;
        geoReg_trainLoss(trId,regId) = info.trainLoss(end);
        geoReg_testLoss(trId,regId) = info.testLoss(end);
        geoReg_generalization(trId,regId) = info.testLoss(end)-info.trainLoss(end);
    end
     paras = rmfield(paras,'G');
     save(['gr_',num2str(numTrain),'.mat'],'net','info','geoReg_acc','geoReg_trainLoss','geoReg_testLoss','geoReg_generalization');
   end
    
   if runWD
    regId = 0;
    paras.lambda = zeros(1,8);
    for  w = [0 1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1:1e-1:5e-1]
        regId = regId+1;
        paras.w = w;
        [net,info] = softmaxTrain(trainX,trainLabel,net0,paras);
        weightDecay_acc(trId,regId) = info.testAcc;
        weightDecay_trainLoss(trId,regId) = info.trainLoss(end);
        weightDecay_testLoss(trId,regId) = info.testLoss(end);
        weightDecay_generalization(trId,regId) = info.testLoss(end)-info.trainLoss(end);
    end
    save(['wd_',num2str(numTrain),'.mat'],'net','info','weightDecay_acc','weightDecay_trainLoss','weightDecay_testLoss','weightDecay_generalization'); 
   end
    
end

