function accu_nnc = nnc(trainX,trainLabel,testX,testLabel,dist)

if nargin==4
    dist = 'euclidean';
end
sizTrainX = size(trainX);
sizTestX = size(testX);
if length(sizTrainX)>2
    trainX = reshape(trainX,[],sizTrainX(end));
end
if length(sizTestX)>2
    testX = reshape(testX,[],sizTestX(end));
end

if (size(trainLabel,1)==1 && size(testLabel,2)==1) || (size(trainLabel,2)==1 && size(testLabel,1)==1)
    trainLabel = trainLabel';
end
id = knnsearch(gather(trainX'),gather(testX'),'Distance',dist);
inferLabel = trainLabel(id);
accu_nnc = sum(inferLabel==testLabel)/length(testLabel);