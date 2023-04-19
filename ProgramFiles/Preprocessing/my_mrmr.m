function [X_train, ca_XY_test, MRMR] = my_mrmr(X_train, y_test, ca_XY_test)
arguments
    X_train (:,:) double
    y_test 
    ca_XY_test = []
end

[idx,mrmr_scores] = fscmrmr(X_train,y_test);
% figure
% bar(mrmr_scores(idx))
% xlabel('Predictor rank')
% ylabel('Predictor importance score')
MRMR.scores = mrmr_scores;
MRMR.limit = 0.15;
X_train = X_train(:, mrmr_scores > MRMR.limit);

n = numel(ca_XY_test);
for i = 1:n
    X_test = ca_XY_test{i}.X(:,1:end-1);
    X_test = X_test(:, mrmr_scores > MRMR.limit);
    X_test(:,end+1) = ca_XY_test{i}.X(:,end);
    ca_XY_test{i}.X = X_test;
end

end

