function accu_nsc = nsc(trainX,trainLabel,testX,testLabel)

N = max(trainLabel);
for i=1:N
    resid = testX-trainX(:,trainLabel==i)*(trainX(:,trainLabel==i)\testX);
    resid_norm(i,:) = sum(resid.^2,1);
end
[~,l_hat] = min(resid_norm,[],1);
if size(testLabel,2)==1
    testLabel = testLabel';
end
accu_nsc = sum(l_hat==testLabel)/length(testLabel);

