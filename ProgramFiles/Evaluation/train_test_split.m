function [X_train, X_test, y_train, y_test] = train_test_split(X,y,train_size)
%TRAIN_TEST_SPLIT 
arguments
    X,y,
    train_size=0.7 % train: 70 %, test: 30 %
end

cv = cvpartition(size(X,1),'HoldOut',1-train_size);

% Separate to training and test data
X_train = X(cv.training,:);
X_test  = X(cv.test,:);
y_train = y(cv.training,:);
y_test = y(cv.test,:);

end

