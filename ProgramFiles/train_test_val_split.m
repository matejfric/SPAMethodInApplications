function [X_train, X_val, X_test, y_train, y_val, y_test] = train_test_val_split(X, y, tr_sz, val_sz)
% TRAIN_TEST_VAL_SPLIT Splits the data into training, validation, and test sets.
%
% Inspired by: 
% https://copyprogramming.com/howto/matlab-split-into-train-valid-test-set-and-keep-proportion
%
arguments
    X
    y
    tr_sz  = 0.7 % train set size
    val_sz = 0.2 % validation set size
end
    % Join the data and labels
    Xy = horzcat(X,y);
    
    % Shuffle the data and labels
    idx = randperm(size(Xy, 1));
    Xy = Xy(idx, :);
    
    % Split data by class
    class_0 = Xy(y == 0, :);
    class_1 = Xy(y == 1, :);
    
    % Train set
    split_tr_0 = round(length(class_0)*tr_sz);
    split_tr_1 = round(length(class_1)*tr_sz);
    train_0 = class_0(1:split_tr_0,:);
    train_1 = class_1(1:split_tr_1,:);
    train_set = vertcat(train_0, train_1);
    train_set = train_set(randperm(length(train_set)),:);
    X_train = train_set(:,1:end-1);
    y_train = train_set(:,end);
    
    % Validation set
    split_valid_0 = split_tr_0 + round(length(class_0)*val_sz);
    split_valid_1 = split_tr_1 + round(length(class_1)*val_sz);
    valid_0 = class_0(split_tr_0+1:split_valid_0,:);
    valid_1 = class_1(split_tr_1+1:split_valid_1,:);
    valid_set = vertcat(valid_0, valid_1);
    valid_set = valid_set(randperm(length(valid_set)),:);
    X_val = valid_set(:,1:end-1);
    y_val = valid_set(:,end);
    
    % Test set
    test_0 = class_0(split_valid_0+1:end,:);
    test_1 = class_1(split_valid_1+1:end,:);
    test_set = vertcat(test_0, test_1);
    test_set = test_set(randperm(length(test_set)),:);
    X_test = test_set(:,1:end-1);
    y_test = test_set(:,end);
end

