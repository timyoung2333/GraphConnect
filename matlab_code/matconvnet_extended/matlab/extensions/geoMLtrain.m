function [net,info] = geoMLtrain(X,trainLabel,net,paras,doVal)
% Deep metric learning, with hinge loss on squared Euclidean distances


rng(0);

if ~isfield(paras,'lambda')
    paras.lambda = 0.5;
end
if ~isfield(paras,'eta')
    paras.eta = 0.01;
end

if ~isfield(paras,'B')
    paras.B = 100; % number of pairs processed each iterate
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

if ~isfield(paras,'regen') % regenerate pairs every 'paras.regen' epochs
    paras.regen = paras.E+1; % if not given, then do not regenerate pairs
end

if ~isfield(paras,'pair')
    paras.pair = 'rand'; % randomly select pairs
end

if strcmp(paras.pair,'rand') 
    if ~isfield(paras,'npos')
        paras.npos = 1;
    end
    if ~isfield(paras,'nneg')
        paras.nneg = 1;
    end
end

if strcmp(paras.pair,'margin') % select "hard" pairs that may violate the margin
    if ~isfield(paras,'npos') 
        paras.npos = 5;% hard positive neighbors each sample
    end
    if ~isfield(paras,'nneg')
        paras.nneg = 10;% hard negative neighbors each sample
    end
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
    for i=1:numel(net.layers)
        if ~isfield(net.layers{i},'filters')
            continue; 
        end
        net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
                    'like', net.layers{i}.filters) ;
        net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
                    'like', net.layers{i}.biases) ; 
        
    end
    info.trainObj = [];
    info.trainACC = [];
    E0 = 0; % start from epoch 1
else
    fprintf('loading saved epoch\n');
    load(paras.load);
    findEpochNum = find(paras.load(1:end-4)=='_',1,'last');
    E0 = str2double(paras.load(findEpochNum+1:end-4));
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

if paras.GPU
    net = vl_simplenn_move_extended(net, 'gpu') ;
    X = gpuArray(X) ;
end

fprintf('generating pairs ...')
if strcmp(paras.pair,'rand')
    pairs = selectPair_rand(trainLabel,paras.npos,paras.nneg);
elseif strcmp(paras.pair,'margin')
    pairs = selectPair_maxmin(X,trainLabel,paras.npos,paras.nneg);
elseif exist(paras.pair, 'file')
    load(paras.pair);
end
fprintf('done\n');
N = size(X,4);
pairSiz = length(pairs);
tau = findTau(X,pairs,trainLabel);
G = learnGraph(X,trainLabel);

%---------------------------------------------------------------
for e=E0+1:E0+paras.E % each epoch randomly permutes the pairs
%---------------------------------------------------------------
 lr = paras.eta(min(e-E0, numel(paras.eta))) ;
 info.trainObj(end+1) = 0 ;
 fprintf('Epoch %02d: batch ', e) ;    
 pairs = pairs(:,randperm(pairSiz));
 
  for b=1:paras.B+1
      
    fprintf('%4d',b);
    pairBatch = pairs(:,(b-1)*floor(pairSiz/paras.B)+1:min(pairSiz,b*floor(pairSiz/paras.B)));
    if isempty(pairBatch), break; end
    i = pairBatch(1,:);
    j = pairBatch(2,:);
    
    im1 = X(:,:,:,i);
    im2 = X(:,:,:,j);
    pLabels = (trainLabel(i)==trainLabel(j))-(trainLabel(i)~=trainLabel(j));
    [res1, res2, objective_thisBatch] = vl_geoMLfb(net, im1, im2, paras.lambda, tau, pLabels, full(G(sub2ind([N N],i,j))));
    
    % gradient step
    for l=1:numel(net.layers)
      if ~isfield(net.layers{l}, 'filters'), continue ; end
        net.layers{l}.filtersMomentum = ...
            paras.m * net.layers{l}.filtersMomentum ...
            - lr * paras.w * net.layers{l}.filters ...
            - lr / length(i) * (res1(l).dzdw{1}+res2(l).dzdw{1}) ;
        
        net.layers{l}.biasesMomentum = ...
            paras.m * net.layers{l}.biasesMomentum ...
            - lr * paras.w * net.layers{l}.biases ...
            - lr / length(i) * (res1(l).dzdw{2}+res2(l).dzdw{2});
      
        net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
        net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
    
    end
    info.trainObj(end) =  info.trainObj(end) + gather(objective_thisBatch)/pairSiz;
    
    fprintf('\b\b\b\b'); % overwrite the displayed batch index
  end % next batch
    
  clear res1 res2;
  fprintf('  objective %.4e', info.trainObj(end));
  
  
  
  % validation
  if doCflag && rem(e-E0,doVal.intev)==0
      
      res = vl_simplenn_extended(net,X,[],[],'conserveMemory',true);
      transed_train = squeeze(res(end).x);
      
      numRight = 0;
      for testB = 1:doVal.B:testSiz
          id = testB:min(testB+doVal.B-1, testSiz);
          res = vl_simplenn_extended(net,doVal.testX(:,:,:,id),[],[],'conserveMemory',true);
          transed_test = squeeze(res(end).x);
          accu = nnc(transed_train, trainLabel, transed_test, doVal.testLabel(id));
          numRight = numRight+accu*length(id);
      end
      info.testACC(end+1) = numRight/testSiz;
      fprintf('  testACC: %.2f%%\n', 100*info.testACC(end));
      
  else
      fprintf('\n');
  end
  
   if rem(e-E0,paras.regen)==0
      fprintf('regenerate pairs ...');
      if strcmp(paras.pair,'rand')
          pairs = selectPair_rand(X,trainLabel,paras.npos,paras.nneg);
      elseif strcmp(paras.pair,'margin')
          pairs = selectPair_maxmin(X,trainLabel,paras.npos,paras.nneg);
      elseif exist(paras.pair, 'file')
          load(paras.pair);
      end
      pairSiz = length(pairs);
      fprintf('done\n');
      tau = findTau(X,pairs,trainLabel);
      
  end
   
  if doSAVE && rem(e-E0,paras.savestep)==0
      save([paras.save,num2str(e),'.mat'],'info','net','lr');
  end
  
end

res = vl_simplenn_extended(net,X,[],[],'conserveMemory',true);
info.fx = squeeze(gather(res(end).x));


% for a study of generalization error
pos = trainLabel(pairs(1,:))==trainLabel(pairs(2,:));
pos_dist = sum((info.fx(:,pairs(1,pos))-info.fx(:,pairs(2,pos))).^2,1);
neg = trainLabel(pairs(1,:))~=trainLabel(pairs(2,:));
neg_dist = sum((info.fx(:,pairs(1,neg))-info.fx(:,pairs(2,neg))).^2,1);
info.trainLoss = (sum(max(0,pos_dist-tau))+sum(max(0,tau-neg_dist)))/pairSiz;

rng(0);
testPairs = selectPair_rand(doVal.testLabel,1000,1000);


testY_pos1 = vl_simplenn_extended(net,doVal.testX(:,:,:,testPairs(1,1:1000)),[],[],'conserveMemory',true);
testY_pos1 = gather(squeeze(testY_pos1(end).x));
testY_pos2 = vl_simplenn_extended(net,doVal.testX(:,:,:,testPairs(2,1:1000)),[],[],'conserveMemory',true);
testY_pos2 = gather(squeeze(testY_pos2(end).x));
pos_dist = sum((testY_pos1-testY_pos2).^2,1);

testY_neg1 = vl_simplenn_extended(net,doVal.testX(:,:,:,testPairs(1,1001:2000)),[],[],'conserveMemory',true);
testY_neg1 = gather(squeeze(testY_neg1(end).x));
testY_neg2 = vl_simplenn_extended(net,doVal.testX(:,:,:,testPairs(2,1001:2000)),[],[],'conserveMemory',true);
testY_neg2 = gather(squeeze(testY_neg2(end).x));
neg_dist = sum((testY_neg1-testY_neg2).^2,1);
info.testLoss = (sum(max(0,pos_dist-tau))+sum(max(0,tau-neg_dist)))/length(testPairs);




%%
function pairs = selectPair_maxmin(X,labels,npos,nneg)
    pairs = [];
    dat = vl_simplenn_extended(net,X,[],[],'conserveMemory',true);
    X = gather(squeeze(dat(end).x));
    for c=unique(labels)
        classi = find(labels==c);
        classNoti = setdiff(1:length(labels),classi);
        
        if length(classi)>1
            k = min(npos,length(classi)-1);
            [~,SAME] = pdist2(X(:,classi)',X(:,classi)','euclidean','Largest',k);
            pairs = [pairs,[reshape(classi(SAME),1,numel(SAME));kron(classi,ones(1,k))]];
        end
        [~,DIFF] = pdist2(X(:,classNoti)',X(:,classi)','euclidean','Smallest',nneg);
        pairs = [pairs,[reshape(classNoti(DIFF),1,numel(DIFF));kron(classi,ones(1,nneg))]];
    end
end

function pairs = selectPair_rand(labels,npos,nneg)
    N = length(labels);
    pairIndicator = bsxfun(@eq, labels, labels');
    pairIndicator(1:N+1:end)=0;
    pairIndicator = squareform(pairIndicator,'tovector');
    posPairs = find(pairIndicator); 
    if npos<=1
        npos = round(numel(posPairs)*npos);
    end
    posPairs = posPairs(randperm(numel(posPairs),npos));
    negPairs = find(~pairIndicator); 
    if nneg<=1
        nneg = round(numel(negPairs)*nneg);
    end
    negPairs = negPairs(randperm(numel(negPairs),nneg));
    
    i = ceil(N-1/2-sqrt((N-1/2)^2-2*posPairs));
    j = posPairs-(N-(i+1)/2).*i+N;
    pairs = [i;j];

    i = ceil(N-1/2-sqrt((N-1/2)^2-2*negPairs));
    j = negPairs-(N-(i+1)/2).*i+N;
    
    pairs = [pairs,[i;j]];
end

function tau = findTau(X,pairs,labels)    
    dat = vl_simplenn_extended(net,X,[],[],'conserveMemory',true);
    X = gather(squeeze(dat(end).x));
    pairIndic = labels(pairs(1,:))==labels(pairs(2,:));
    dist_pos = sum((X(:,pairs(1,pairIndic==1))-X(:,pairs(2,pairIndic==1))).^2,1);
    dist_neg = sum((X(:,pairs(1,pairIndic==0))-X(:,pairs(2,pairIndic==0))).^2,1);
    tau = 0.5*(mean(dist_pos)+mean(dist_neg));
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

end








