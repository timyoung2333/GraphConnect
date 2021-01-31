function [net, info, G] = softmaxTrain_geoReg(trainX,trainLabel,net,paras)
% implement a deep convolutional or fully connected network that has same
% spirit as deepface
% The top layer is a softmaxloss layer by default

rng(0);

if ~isfield(paras,'eta')
    paras.eta = 0.01;
end

if ~isfield(paras,'B')
    paras.B = 100;
end

if ~isfield(paras,'E')
    paras.E = 50;
end

if ~isfield(paras,'w')
    paras.w = 0;
end

if ~isfield(paras,'lambda')
    paras.lambda = zeros(1,length(net.layers));
end

if ~isfield(paras,'m')
    paras.m = 0.9;
end

if ~isfield(paras,'GPU')
    paras.GPU = 0;
end


if ~isfield(paras,'loadmodel')
    doLOAD = 0;
    info.trainLoss = [];
    info.trainAcc = [];
    E0 = 0; % start from epoch 1
else
    doLOAD = 1;
    fprintf('loading saved epoch\n'); % the saved epoch include model, loss and accuracy
    load(paras.loadmodel);
    findEpochNum = find(paras.loadmodel(1:end-4)=='_',1,'last'); % saved epoch file name must be in the form of *_epoch number.mat
    E0 = str2double(paras.loadmodel(findEpochNum+1:end-4));
end

if ~isfield(paras,'saveprefix')
    doSAVE = 0;
else
    doSAVE = 1;
    if ~isfield(paras,'snapshot')
        paras.snapshot = paras.E; % only save the last epoch
    end
end

if isfield(paras,'testdat') && isfield(paras,'testlabel')
    doTEST = 1;
    info.testAcc = [];
    info.testLoss = [];
    if size(paras.testlabel,2)>1
        paras.testlabel = paras.testlabel';
    end
    if ~isfield(paras,'teststep')
        paras.teststep = 1;
    end
else
    doTEST = 0;
end


% Initalize momentum, learning rate multiplier, weightDecay multiplier, etc.
if doLOAD==0
    for i=1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            J = numel(net.layers{i}.weights) ;
            for j=1:J
                net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end


if paras.GPU
  net = vl_simplenn_move_extended(net, 'gpu') ;
  trainX = gpuArray(trainX);
  if doTEST
        paras.testdat = gpuArray(paras.testdat);
  end
end

if ~isfield(paras,'G')
    G = learnGraph(trainX,trainLabel);
else 
    G = paras.G;
end

% some global constant
trainSiz = numel(trainLabel);
if doTEST
    testSiz = numel(paras.testlabel);
end
if size(trainLabel,2)>1
    trainLabel = trainLabel';
end
if paras.GPU
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------


for e=E0+1:E0+paras.E
  lr = paras.eta(min(e-E0, numel(paras.eta))) ;

  train = randperm(trainSiz) ;

  info.trainLoss(end+1) = 0 ;
  info.trainAcc(end+1) = 0 ;


  fprintf('Epoch %02d: batch ', e) ;             
  
  for t=1:paras.B:trainSiz
    % get next image batch and labels
    batch = train(t:min(t+paras.B-1, trainSiz)) ;
    batchSize = length(batch);
    fprintf('%4d',fix(t/paras.B)+1);
    

    % backprop
    net.layers{end}.class = trainLabel(batch) ;
    res = vl_simplenn_extended_geoReg(net, trainX(:,:,:,batch), one, [], full(computeL(G(batch,batch))),paras.lambda) ;
    info.trainLoss(end) = info.trainLoss(end) + double(gather(res(end).x)) ;
    [~,inferLabels_train] = max(gather(res(end-1).x),[],3);
    info.trainAcc(end) = info.trainAcc(end)+sum(squeeze(inferLabels_train)==trainLabel(batch));

    for l=1:numel(net.layers)
	for j=1:numel(res(l).dzdw)
            thisDecay = paras.w * net.layers{l}.weightDecay(j) ;
            thisLR = lr * net.layers{l}.learningRate(j) ;
            
            if isfield(net.layers{l}, 'weights')
                net.layers{l}.momentum{j} = ...
                    paras.m * net.layers{l}.momentum{j} ...
                    - thisDecay * net.layers{l}.weights{j} ...
                    - (1 / batchSize) * res(l).dzdw{j} ;
                net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;
            end
        end
    end   
    
    fprintf('\b\b\b\b');
    clear res;
  end % next batch

  info.trainLoss(end) = info.trainLoss(end)/trainSiz ;
  info.trainAcc(end) = info.trainAcc(end)/trainSiz;
  fprintf('\b\b\b\b\b\b  train loss: %4f  train acc: %.2f%%', info.trainLoss(end), info.trainAcc(end)*100);
  
  % validation
  if doTEST && rem(e-E0,paras.teststep)==0
      infer_testLabel = []; 
      info.testLoss(end+1)=0;
      for testB = 1:paras.B:testSiz
          id = testB:min(testB+paras.B-1, testSiz);
          net.layers{end}.class = paras.testlabel(id) ;
          res_test = vl_simplenn_extended(net,paras.testdat(:,:,:,id),[],[], 'conserveMemory', false, 'sync', true);
          [~,infer_testLabel(:,:,:,id)] = max(gather(res_test(end-1).x),[],3);
          info.testLoss(end)=info.testLoss(end)+gather(double(res_test(end).x));
      end
      infer_testLabel = squeeze(infer_testLabel);
      info.testAcc(end+1) = sum(infer_testLabel==paras.testlabel)/testSiz;
      info.testLoss(end) = info.testLoss(end)/testSiz;
      fprintf('  test loss: %.4f  test acc: %.2f%%\n', info.testLoss(end), 100*info.testAcc(end));
      clear res_test;
  else
      fprintf('\n');
  end
  
  
  if doSAVE && rem(e-E0,paras.snapshot)==0
      save([paras.saveprefix,'_',num2str(e),'.mat'],'info','net');
  end
  
end


function G = learnGraph(X,labels)
    N = size(X,4);
    X = gather(reshape(X,[],N)');
    G = sparse(zeros(N));
    for c=unique(labels)
        D = pdist(X(labels==c,:));
        G(labels==c,labels==c) = squareform(exp(-(D/mean(D)).^2));%inf
    end
end

function L = computeL(G)
    L = diag(sum(G,2))-G;
end
end
