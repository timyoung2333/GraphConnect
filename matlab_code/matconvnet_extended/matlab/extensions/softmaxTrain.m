function [net, info] = softmaxTrain(trainX,trainLabel,net,paras,doVal)
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

if ~isfield(paras,'m')
    paras.m = 0.9;
end

if ~isfield(paras,'GPU')
    paras.GPU = 0;
end


if ~isfield(paras,'save') % don't save
    doSAVE = 0;
else % epoch file prefix given
    doSAVE = 1;
    if ~isfield(paras,'savestep')
	paras.savestep = paras.E; % only save the last epoch
    end
end

if ~isfield(paras, 'load') % do not load
    info.trainObj = [];
    info.trainACC = [];
    E0 = 0; % start from epoch 1
else
    fprintf('loading saved epoch\n');
    load(paras.load,'net','info');
    findEpochNum = find(paras.load(1:end-4)=='_',1,'last');
    E0 = str2double(paras.load(findEpochNum+1:end-4));
end

% initialize for iterations
for i=1:numel(net.layers)
      if ~isfield(net.layers{i},'filters')
          continue; 
      end
      if ~isfield(net.layers{i},'filtersMomentum')
      	  net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
                    'like', net.layers{i}.filters) ;
      end
      if ~isfield(net.layers{i},'biasesMomentum')
	  net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
                    'like', net.layers{i}.biases) ; 
      end
      	
      if ~isfield(net.layers{i}, 'filtersLearningRate')
    	  net.layers{i}.filtersLearningRate = 1 ;
      end
      if ~isfield(net.layers{i}, 'biasesLearningRate')
   	  net.layers{i}.biasesLearningRate = 1 ;
      end
      if ~isfield(net.layers{i}, 'filtersWeightDecay')
    	  net.layers{i}.filtersWeightDecay = 1 ;
      end
      if ~isfield(net.layers{i}, 'biasesWeightDecay')
   	  net.layers{i}.biasesWeightDecay = 1 ;
      end
end
   


if exist('doVal','var')
    doCflag = 1;
    if ~isfield(info,'testACC')
        info.testACC = [];
    end
    if ~isfield(info,'testObj')
        info.testObj = [];
    end
    testSiz = length(doVal.testLabel);
    if paras.GPU
        doVal.testX = gpuArray(doVal.testX);
    end
    if size(doVal.testLabel,2)>1
        doVal.testLabel = doVal.testLabel';
    end
else
    doCflag = 0;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
if paras.GPU
  net = vl_simplenn_move_extended(net, 'gpu') ;
  trainX = gpuArray(trainX);
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------


trainSiz = numel(trainLabel);
if size(trainLabel,2)>1
    trainLabel = trainLabel';
end
if paras.GPU
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

lr = 0;
for e=E0+1:E0+paras.E
    if lr~=paras.eta(min(e-E0, numel(paras.eta)))
        for l=1:numel(net.layers)
            if ~isfield(net.layers{l}.type, 'filters'), continue ; end
            net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
            net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
        end
    end
        
  lr = paras.eta(min(e-E0, numel(paras.eta))) ;

  train = randperm(trainSiz) ;

  info.trainObj(end+1) = 0 ;
  info.trainACC(end+1) = 0 ;


  fprintf('Epoch %02d: batch ', e) ;             
  
  for t=1:paras.B:trainSiz
    % get next image batch and labels
    batch = train(t:min(t+paras.B-1, trainSiz)) ;
    fprintf('%4d',fix(t/paras.B)+1);
    
    im_thisBatch = trainX(:,:,:,batch);

    % backprop
    net.layers{end}.class = trainLabel(batch) ;
    res = vl_simplenn_extended(net, im_thisBatch, one, [], 'conserveMemory', false, 'sync', true) ;
    info.trainObj(end) = info.trainObj(end) + double(gather(res(end).x)) ;
    [~,inferLabels_train] = max(gather(res(end-1).x),[],3);
    info.trainACC(end) = info.trainACC(end)+sum(squeeze(inferLabels_train)==trainLabel(batch));

    % gradient step
    for l=1:numel(net.layers)
      if ~isfield(net.layers{l},'filters'), continue ; end

      net.layers{l}.filtersMomentum = ...
        paras.m * net.layers{l}.filtersMomentum ...
          - (lr*net.layers{l}.filtersLearningRate) * (paras.w*net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
          - (lr*net.layers{l}.filtersLearningRate) / length(batch) * res(l).dzdw{1} ;

      net.layers{l}.biasesMomentum = ...
        paras.m * net.layers{l}.biasesMomentum ...
          - (lr*net.layers{l}.biasesLearningRate) * (paras.w*net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
          - (lr*net.layers{l}.biasesLearningRate) / length(batch) * res(l).dzdw{2} ;

      net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
      net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
    end
    
    
    fprintf('\b\b\b\b');
    clear res;
  end % next batch

  info.trainObj(end) = info.trainObj(end)/trainSiz ;
  info.trainACC(end) = info.trainACC(end)/trainSiz;
  fprintf('\b\b\b\b\b\b  trainObj: %4f  trainACC: %.2f%%', info.trainObj(end), info.trainACC(end)*100);
  
  % validation
  if doCflag && rem(e-E0,doVal.intev)==0
      infer_testLabel = []; 
      info.testObj(end+1)=0;
      for testB = 1:doVal.B:testSiz
          id = testB:min(testB+doVal.B-1, testSiz);
          net.layers{end}.class = doVal.testLabel(id) ;
          res_test = vl_simplenn_extended(net,doVal.testX(:,:,:,id),[],[], 'conserveMemory', false, 'sync', true);
          [~,infer_testLabel(:,:,:,id)] = max(gather(res_test(end-1).x),[],3);
          info.testObj(end)=info.testObj(end)+gather(double(res_test(end).x));
      end
      infer_testLabel = squeeze(infer_testLabel);
      info.testACC(end+1) = sum(infer_testLabel==doVal.testLabel)/testSiz;
      info.testObj(end) = info.testObj(end)/testSiz;
      fprintf('  testObj: %.4f  testACC: %.2f%%\n', info.testObj(end), 100*info.testACC(end));
      clear res_test;
  else
      fprintf('\n');
  end
  
  
  if doSAVE && rem(e-E0,paras.savestep)==0
      save([paras.save,'_',num2str(e),'.mat'],'info','net');
  end
  
end

net.layers(end) = [];
res = vl_simplenn_extended(net,trainX);
info.fx = reshape(res(end-1).x,[],trainSiz);
