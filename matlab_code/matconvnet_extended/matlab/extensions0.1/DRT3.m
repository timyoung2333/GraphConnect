function [net,info] = DRT3(X,trainLabel,net,paras)
% deep discriminative-robust transform
% metric learning objective + local robustness constraint



if ~isfield(paras,'eta')
    paras.eta = 0.01;
end

if ~isfield(paras,'taustep')
    paras.taustep = 0;
end

if ~isfield(paras,'lambda')
    paras.lambda = 0.5;
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

if ~isfield(paras,'kn')
    paras.kn = inf;
end

if ~isfield(paras,'newpair') % regenerate pairs every 'paras.regen' epochs
    paras.newpair = paras.E+1; % if not given, then do not regenerate pairs
end


if ~isfield(paras,'npos')
    paras.npos = 1;
end
if ~isfield(paras,'nneg')
    paras.nneg = 1;
end


if ~isfield(paras,'GPU')
    paras.GPU = 0;
end

if ~isfield(paras,'loadmodel')
    doLOAD = 0;
    info.trainLoss = [];
    info.trainIntraDists = [];
    info.trainInterDists = [];
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
    info.testIntraDists = [];
    info.testInterDists = [];
    if size(paras.testlabel,1)>1
        paras.testlabel = paras.testlabel';
    end
    if ~isfield(paras,'teststep')
        paras.teststep = 1;
    end
    N_test = length(paras.testlabel);
    testpairLabels = bsxfun(@eq,paras.testlabel,paras.testlabel'); 
    testpairLabels = single(testpairLabels);
    testpairLabels(testpairLabels==0)=-1;
    testpairLabels(1:N_test+1:end)=0;
    testpairLabels=squareform(testpairLabels,'tovector');
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
    X = gpuArray(X) ;
    if doTEST
        paras.testdat = gpuArray(paras.testdat);
    end
end

% some global constant
if doTEST
    testSiz = numel(paras.testlabel);
end
if size(trainLabel,1)>1
    trainLabel = trainLabel'; % make class label a row vector
end


fprintf('generating pairs ...')
[pairs, pairLabels] = selectPair_rand(trainLabel,paras.npos,paras.nneg);
regu = computeRegularizer(X,pairs,trainLabel);
tau = findTau(X); info.tau = tau;
pairSiz = length(pairs);
fprintf('done\n');


rng(0);
%---------------------------------------------------------------
for e=E0+1:E0+paras.E % each epoch randomly permutes the pairs
%---------------------------------------------------------------
 lr = paras.eta(min(e-E0, numel(paras.eta))) ;
 info.trainLoss(end+1) = 0 ;
 fprintf('Epoch %02d: batch ', e) ; 
 permInd = randperm(pairSiz);
 pairs = pairs(:,permInd);
 pairLabels = pairLabels(permInd);
 regu = regu(permInd);
 
  for b=1:paras.B+1
      
    fprintf('%4d',b);
    pairBatch = pairs(:,(b-1)*floor(pairSiz/paras.B)+1:min(pairSiz,b*floor(pairSiz/paras.B)));
    r = regu((b-1)*floor(pairSiz/paras.B)+1:min(pairSiz,b*floor(pairSiz/paras.B)));
    pL = pairLabels((b-1)*floor(pairSiz/paras.B)+1:min(pairSiz,b*floor(pairSiz/paras.B)));
    batchSize = size(pairBatch,2);
    if isempty(pairBatch), break; end
    [res1, res2, dtau] = vl_DRTfb(net, X(:,:,:,pairBatch(1,:)), X(:,:,:,pairBatch(2,:)), tau, pL, r, paras.lambda);
    
    
    % gradient step
    tau = tau-dtau/batchSize*paras.taustep;
    
    for l=1:numel(net.layers)
        for j=1:numel(res1(l).dzdw)
            thisDecay = paras.w * net.layers{l}.weightDecay(j) ;
            thisLR = lr * net.layers{l}.learningRate(j) ;
            
            if isfield(net.layers{l}, 'weights')
                net.layers{l}.momentum{j} = ...
                    paras.m * net.layers{l}.momentum{j} ...
                    - thisDecay * net.layers{l}.weights{j} ...
                    - (res1(l).dzdw{j} + res2(l).dzdw{j}) ;
                net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;
            end
        end
        
    end
      
    fprintf('\b\b\b\b'); % overwrite the displayed batch index
    
    
  end % next batch
  info.tau = [info.tau,tau];
  res = vl_simplenn_extended(net,X,[],[],'conserveMemory',true);
  transed_train = squeeze(res(end).x);
  [info.trainLoss(e),  info.trainCost(e), info.trainIntraDists(e), info.trainInterDists(e)] = computeTrainLoss(transed_train);
  
  clear res1 res2;
  fprintf('  trainCost %.4e  trainLoss %.4e', info.trainCost(end), info.trainLoss(end));
  
  
  % validation
  if doTEST && rem(e-E0,paras.teststep)==0    
      numRight = 0;
      for b = 1:paras.B+1
          id = (b-1)*floor(testSiz/paras.B)+1:min(testSiz,b*floor(testSiz/paras.B));
          if isempty(id), break; end
          res_test = vl_simplenn_extended(net,paras.testdat(:,:,:,id),[],[],'conserveMemory',true);
          transed_test(:,id) = squeeze(res_test(end).x);
          accu = nnc(transed_train, trainLabel, transed_test(:,id), paras.testlabel(id)); % use nearest neighbor classifier to validate
          numRight = numRight+accu*length(id);
      end
      info.testAcc(end+1) = numRight/testSiz;
      [info.testLoss(end+1), info.testIntraDists(end+1), info.testInterDists(end+1)]= computeTestLoss(transed_test);
      fprintf('  testLoss %.4e  testAcc: %.2f%%\n', info.testLoss(end), 100*info.testAcc(end));
      
  else
      fprintf('\n');
  end
  
   if rem(e-E0,paras.newpair)==0
      fprintf('regenerate pairs ...');
      [pairs, pairLabels] = selectPair_rand(trainLabel,paras.npos,paras.nneg);
      pairSiz = length(pairs);
      regu = computeRegularizer(X,pairs,trainLabel,paras.lambda);
      fprintf('done\n');
   end
   
  if doSAVE && rem(e-E0,paras.snapshot)==0
      save([paras.saveprefix,num2str(e),'.mat'],'info','net');
  end
  
end




%%

function [pairs, pairLabels] = selectPair_rand(labels,npos,nneg)
    rng(0);
    N = length(labels);
    pairIndicator = bsxfun(@eq, labels, labels');
    pairIndicator(1:N+1:end)=0;
    pairIndicator = squareform(pairIndicator,'tovector');
    posPairs = find(pairIndicator); 
    if npos<1
        npos = round(numel(posPairs)*npos);
        posPairs = posPairs(randperm(numel(posPairs),npos));
    elseif npos>1
        posPairs = posPairs(randperm(numel(posPairs),npos));
    end
    negPairs = find(~pairIndicator); 
    if nneg<1
        nneg = round(numel(negPairs)*nneg);
        negPairs = negPairs(randperm(numel(negPairs),nneg));
    elseif nneg>1
        negPairs = negPairs(randperm(numel(negPairs),nneg));
    end    
    
    
    i = ceil(N-1/2-sqrt((N-1/2)^2-2*posPairs));
    j = posPairs-(N-(i+1)/2).*i+N;
    pairs = [i;j];
    pairLabels = ones(1,length(i));
    
    i = ceil(N-1/2-sqrt((N-1/2)^2-2*negPairs));
    j = negPairs-(N-(i+1)/2).*i+N;
    
    pairs = [pairs,[i;j]];
    pairLabels = [pairLabels,-ones(1,length(i))];
    
end

function dists = compute_pdist(dat,pairs)
    if length(size(dat))==2
       p1 = dat(:,pairs(1,:));
       p2 = dat(:,pairs(2,:));
    elseif length(size(dat))==4
       p1 = reshape(dat(:,:,:,pairs(1,:)),[],length(pairs(1,:)));
       p2 = reshape(dat(:,:,:,pairs(2,:)),[],length(pairs(2,:)));
    end
    dists = gather(sqrt(sum((p1-p2).^2,1)));
end

function regu = computeRegularizer(X,pairs,trainLabel)
    regu = sparse(zeros(size(pairLabels)));
    if paras.kn==inf
        regu(pairLabels==1) = compute_pdist(X,pairs(:,pairLabels==1));
    else % within each class, find the knn
        neighbors = [];
        for c=unique(trainLabel)
            class_c = find(trainLabel==c);
            Xvec = reshape(X(:,:,:,class_c),[],length(class_c));
            knid = knnsearch(gather(Xvec'),gather(Xvec'),'K',paras.kn+1);
            knid(:,1) = []; % exclude self to self
            neighbors = [neighbors,[kron(ones(1,paras.kn),class_c);class_c(knid(:))]];
        end
        inNeighbor = ismember(pairs',neighbors','rows');
        regu(inNeighbor) = compute_pdist(X,pairs(:,inNeighbor));
    end
end

function tau = findTau(X)    
    dat = vl_simplenn_extended(net,X,[],[],'conserveMemory',true);
    dat = dat(end).x;
    dist_pos = mean(compute_pdist(dat,pairs(:,pairLabels==1)));
    dist_neg = mean(compute_pdist(dat,pairs(:,pairLabels==-1)));
    tau = [dist_pos,dist_neg];
end


function [loss, cost, intraDists, interDists] = computeTrainLoss(X)
    dists = compute_pdist(X,pairs);
    loss = mean([ max( tau(2)-dists(pairLabels==-1), 0), max( dists(pairLabels==1)-tau(1), 0 )  ]);
    cost = loss+paras.lambda*mean(abs( dists(regu~=0)-full(regu(regu~=0)) ));
    
    intraDists = mean(dists(pairLabels==1));
    interDists = mean(dists(pairLabels==-1));
end

function [loss, intraDists, interDists] = computeTestLoss(X)
    gramX = gather(X'*X);
    normX = gather(sum(X.^2,1));
    distsX = sqrt(bsxfun(@plus,bsxfun(@plus,normX,-2*gramX),normX'));
    distsX(1:N_test+1:end) = 0;
    distsX = squareform(gather(distsX),'tovector');
    loss = mean( [max(tau(2)-distsX(testpairLabels==-1),0),max(distsX(testpairLabels==-1)-tau(1),0) ] );
    
    intraDists = mean(distsX(testpairLabels==1));
    interDists = mean(distsX(testpairLabels==-1));
end


function [res1, res2, dtau] = vl_DRTfb(net, x1, x2, tau, pLabels, r, lambda)

res1 = vl_simplenn_extended(net,x1);
res2 = vl_simplenn_extended(net,x2);

%%%% compute gradients in the two replicas %%%%
delta_ij = squeeze(res1(end).x-res2(end).x);
pairDist = sqrt(sum(delta_ij.^2,1));
delta_ij_normcol = bsxfun(@rdivide,delta_ij,pairDist);
neighbors = (r~=0);
r = full(r(neighbors));

dzdy = bsxfun(@times,delta_ij_normcol,pLabels+single(pLabels==-1 & pairDist>tau(2))-single(pLabels==1 & pairDist<tau(1))) / length(pLabels);
dzdy(:,neighbors) = dzdy(:,neighbors)+lambda*bsxfun(@times,sign(pairDist(neighbors)-r),delta_ij_normcol(:,neighbors))/length(r);

dzdy = permute(dzdy,[3 4 1 2]);
dtau = gather([-sum(pLabels==1 & pairDist>tau(1)) , sum(pLabels==-1 & pairDist<tau(2))] )/length(pLabels);

res1 = vl_simplenn_extended(net,res1(1).x,dzdy,res1,'onlyGradient',true);
res2 = vl_simplenn_extended(net,res2(1).x,-dzdy,res2,'onlyGradient',true);

end

end

