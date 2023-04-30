function [] = test_standard_classifiers(X_train,X_test,y_train,y_test)
%TEST_STANDARD_CLASSIFIERS

% KNN
k = 5;
mdlKNN = fitcknn(X_train, y_train, 'NumNeighbors', k);
y_pred = mdlKNN.predict(X_test);
statsKNN = statistics(y_pred, y_test);

%SVM
if sum(unique(y_train))<=2
    mdlSVM = fitcsvm(X_train, y_train);
else
    mdlSVM = fitcecoc(X_train, y_train);
end
y_pred = mdlSVM.predict(X_test);
statsSVM = statistics(y_pred, y_test);

%Decision Tree
mdlTree = fitctree(X_train, y_train,'CrossVal','on');
numBranches = @(x)sum(x.IsBranch);
mdlTreeNumSplits = cellfun(numBranches, mdlTree.Trained);
%view(mdlTree.Trained{1},'Mode','graph')

%Pruned Decision Tree
mdlTreePruned = fitctree(X_train, y_train,'MaxNumSplits',10,'CrossVal','on');
%view(mdlTreePruned.Trained{1},'Mode','graph')
classErrorTreeDefault = kfoldLoss(mdlTree);
classErrorTreePruned = kfoldLoss(mdlTreePruned);
mdlTreePruned = fitctree(X_train, y_train,'MaxNumSplits',10);
y_pred = mdlTreePruned.predict(X_test);
statsTree = statistics(y_pred, y_test);

%Naive Bayes
mdlNB = fitcnb(X_train, y_train);
y_pred = mdlNB.predict(X_test);
statsNB = statistics(y_pred, y_test);

%Discriminant Analysis
mdlDiscr = fitcdiscr(X_train, y_train);
y_pred = mdlDiscr.predict(X_test);
statsDiscr = statistics(y_pred, y_test);

%Generalized linear regression
mdlGLM = fitglm(X_train, y_train);
y_pred = mdlGLM.predict(X_test);
statsGLM = statistics(y_pred, y_test);

fprintf("\n| Standard classifiers          | Accuracy | F1-score |\n")
fprintf("|-------------------------------|----------|----------|\n")
fprintf("| K-Nearest Neighbors           |   %.3f  |   %.3f  |\n", statsKNN.accuracy, statsKNN.f1score)
fprintf("| Support Vector Machine        |   %.3f  |   %.3f  |\n", statsSVM.accuracy, statsSVM.f1score)
fprintf("| Decision Tree (pruned)        |   %.3f  |   %.3f  |\n", statsTree.accuracy, statsTree.f1score)
fprintf("| Discriminant Analysis         |   %.3f  |   %.3f  |\n", statsDiscr.accuracy, statsDiscr.f1score)
fprintf("| Generalized Linear Regression |   %.3f  |   %.3f  |\n", statsGLM.accuracy, statsGLM.f1score)
fprintf("| Naive Bayes                   |   %.3f  |   %.3f  |\n\n", statsNB.accuracy, statsNB.f1score)

end

