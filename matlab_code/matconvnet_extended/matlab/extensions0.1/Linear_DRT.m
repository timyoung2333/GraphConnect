function [A, info] = Linear_DRT(X,trainLabels,paras)

rng(0);
dimX = size(X);
trainSiz = dimX(end);
X = reshape(X,[],trainSiz);
if size(trainLabels,2)>1 trainLabels=trainLabels';end % labels in column vector

if ~isfield(paras,'d')
    paras.d = size(X,1);
end

if ~isfield(paras,'eta')
    paras.eta = 0.01;
end

if ~isfield(paras,'w')
    paras.w = 0;
end

if ~isfield(paras,'taustep')
    paras.taustep = 0;
end

if ~isfield(paras,'lambda')
    paras.lambda = 0.5;
end


if ~isfield(paras,'E')
    paras.E = 100;
end

if ~isfield(paras,'kn')
    paras.kn = inf;
end


if ~isfield(paras,'GPU')
    paras.GPU = 0;
end
if paras.GPU
    X = gpuArray(X);
end

if isfield(paras,'testdat') && isfield(paras,'testlabel')
    doTEST = 1;
    info.testAcc = [];
    info.testLoss = [];
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




% learn neighborhoods
[D, neighborPairsID] = learnNeighbors(X,trainLabels);
neighborPairs = indexMapping(trainSiz,find(neighborPairsID));

% generate pairLabels
pairLabels = bsxfun(@eq,trainLabels,trainLabels');
pairLabels = single(pairLabels);
pairLabels(pairLabels==0) = -1;
pairLabels(1:length(trainLabels)+1:end) = 0;
pairLabels = squareform(pairLabels,'tovector');

% find tau
A = randn(paras.d,size(X,1))/sqrt(paras.d);
tau = findTau(A*X,pairLabels);

info.trainCost = [];
info.trainLoss = [];
info.trainIntraDists = [];
info.trainInterDists = [];
if doTEST
    info.testLoss = [];
    info.testIntraDists = [];
    info.testInterDists = [];
end


for e=1:paras.E
    [info.trainCost(e), info.trainLoss(e), grad, info.trainIntraDists(:,e), info.trainInterDists(:,e)] = trainIt(A); 
    tau = tau-paras.taustep*grad{1};
    A = A-paras.eta*grad{2};
    fprintf('Iteration %3d, trainCost %.4e, trainLoss %4e, ', e, info.trainCost(end), info.trainLoss(end));
    
    if rem(e,paras.teststep)==0
        [info.testLoss(end+1), info.testAcc(end+1), info.testIntraDists(:,end+1), info.testInterDists(:,end+1)] = testIt(A);
        fprintf('testLoss %.4e, testAcc %.2f%%', info.testLoss(end), info.testAcc(end)*100);
    end
    
    fprintf('\n');
end




%%

function sub = indexMapping(siz,ind)
    sub(1,:) = ceil(siz-1/2-sqrt((siz-1/2)^2-2*ind));
    sub(2,:) = ind-(siz-(sub(1,:)+1)/2).*sub(1,:)+siz;
    
    
end

function [D,neighborPairsID] = learnNeighbors(X,trainLabels)
    D = sparse(size(X,2),size(X,2));
    for c=unique(trainLabels')
        class_c = find(trainLabels==c);
        [IDc, dc] = knnsearch(gather(X(:,class_c)'),gather(X(:,class_c)'),'k',paras.kn+1);
        IDc(:,1) = []; dc(:,1) = [];
        IDc = sub2ind(size(D),kron(ones(paras.kn,1),class_c),class_c(IDc(:)));
        D(IDc) = dc;
    end
    D = max(D,D');
    D = sparse(squareform(D.^2,'tovector'));
    neighborPairsID = (D~=0);
    
end

function tau = findTau(dat,pairLabels)
    datDists = pdist(gather(dat')).^2;    
    intra = mean(datDists(pairLabels==1));
    inter = mean(datDists(pairLabels==-1));
    tau = gather(mean([intra,inter]));
    
end


function [cost, loss, grad, intraDists, interDists] = trainIt(A)
    Y = A*X;
    distdat = pdist(gather(Y')).^2;
    intraDists = [mean(distdat(pairLabels==1)); max(distdat(pairLabels==1)); min(distdat(pairLabels==1))];
    interDists = [mean(distdat(pairLabels==-1)); max(distdat(pairLabels==-1)); min(distdat(pairLabels==-1))];    
    
    effPos = find(pairLabels==1 & distdat>tau);
    effNeg = find(pairLabels==-1 & distdat<tau);
    grad{1} = (numel(effNeg)-numel(effPos))/length(pairLabels); % gradient w.r.t. tau
    
    loss = (sum(distdat(effPos)-tau)+sum(tau-distdat(effNeg)))/length(pairLabels);
    cost = loss+paras.lambda*sum(abs(distdat(neighborPairsID)-full(D(neighborPairsID))))/length(pairLabels) + paras.w*norm(A,'fro')^2;
    
    effPos = indexMapping(trainSiz,effPos);
    effNeg = indexMapping(trainSiz,effNeg);
    
    grad_discrimination = (Y(:,effPos(1,:))-Y(:,effPos(2,:))) * (X(:,effPos(1,:))-X(:,effPos(2,:)))'...
                            -(Y(:,effNeg(1,:))-Y(:,effNeg(2,:))) * (X(:,effNeg(1,:))-X(:,effNeg(2,:)))';
                            
    grad_robustness = bsxfun(@times, sign(distdat(neighborPairsID)-full(D(neighborPairsID))), Y(:,neighborPairs(1,:))-Y(:,neighborPairs(2,:)) )...
                            * (X(:,neighborPairs(1,:))-X(:,neighborPairs(2,:)))'; 
                                             
    grad{2} = (grad_discrimination+paras.lambda*grad_robustness)/length(pairLabels)+paras.w*A;
end



function [loss, testAcc, intraDists, interDists] = testIt(A)
    transed_test = gather(A)*paras.testdat;
    testNorm = sum(transed_test.^2,1);
    dists_test = bsxfun(@plus, bsxfun(@plus, testNorm, -2*transed_test'*transed_test), testNorm');    
    dists_test(1:size(paras.testdat,2)+1:end) = 0;
    dists_test = squareform(dists_test,'tovector');
    
    intraDists = [mean(dists_test(testpairLabels==1)); max(dists_test(testpairLabels==1)); min(dists_test(testpairLabels==1))];
    interDists = [mean(dists_test(testpairLabels==-1)); max(dists_test(testpairLabels==-1)); min(dists_test(testpairLabels==-1))];  
    
    loss = mean(max(0,testpairLabels.*(dists_test-tau)));   
    testAcc = nnc(gather(A*X),trainLabels,transed_test,paras.testlabel);
    
end

end






