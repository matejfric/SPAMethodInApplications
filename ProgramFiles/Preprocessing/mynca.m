function [selidx] = mynca(X,y)
%MYNCA Feature selection for classification using
%      neighborhood component analysis (NCA)

cvp = cvpartition(y,'holdout',0.25 * length(y));
Xtrain = X(cvp.training,:);
ytrain = y(cvp.training,:);
Xtest  = X(cvp.test,:);
ytest  = y(cvp.test,:);

%% Determine if feature selection is necessary
nca = fscnca(Xtrain,ytrain,'FitMethod','none');
L = loss(nca,Xtest,ytest)

nca = fscnca(Xtrain,ytrain,'FitMethod','exact','Lambda',0,...
      'Solver','sgd','Standardize',true);
L = loss(nca,Xtest,ytest)

%%
cvp = cvpartition(ytrain,'kfold',5);
numvalidsets = cvp.NumTestSets;
n = length(ytrain);
lambdavals = linspace(0,20,20)/n;
lossvals = zeros(length(lambdavals),numvalidsets);

for i = progress(1:length(lambdavals))
    for k = 1:numvalidsets
        X = Xtrain(cvp.training(k),:);
        y = ytrain(cvp.training(k),:);
        Xvalid = Xtrain(cvp.test(k),:);
        yvalid = ytrain(cvp.test(k),:);

        nca = fscnca(X,y,'FitMethod','exact', ...
             'Solver','sgd','Lambda',lambdavals(i), ...
             'IterationLimit',30,'GradientTolerance',1e-4, ...
             'Standardize',true);
                  
        lossvals(i,k) = loss(nca,Xvalid,yvalid,'LossFunction','classiferror');
    end
end

meanloss = mean(lossvals,2);

figure()
plot(lambdavals,meanloss,'ro-')
xlabel('Lambda')
ylabel('Loss (MSE)')
grid on

[~,idx] = min(meanloss); % Find the index

bestlambda = lambdavals(idx) % Find the best lambda value

bestloss = meanloss(idx)

nca = fscnca(Xtrain,ytrain,'FitMethod','exact','Solver','sgd',...
    'Lambda',bestlambda,'Standardize',true,'Verbose',1);

figure()
plot(nca.FeatureWeights,'ro')
xlabel('Feature index')
ylabel('Feature weight')
grid on

tol    = 0.02;
selidx = find(nca.FeatureWeights > tol*max(1,max(nca.FeatureWeights)))

L = loss(nca,Xtest,ytest)


end

