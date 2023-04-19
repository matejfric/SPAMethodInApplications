function [X_train, X_val, X_test, y_train, y_val, y_test] = ...
    train_test_val_split(X, y, tr_sz, val_sz)
% TRAIN_TEST_VAL_SPLIT 
%
% Splits the data into training, validation, and test sets
% with stratification.
%
% If val_sz = 0 this becomes train_test_split.
%
% Inspired by: 
% https://copyprogramming.com/howto/matlab-split-into-train-valid-test-set-and-keep-proportion
%
arguments
    X (:,:) double  % data
    y (:,1) double  % vector of labels
    tr_sz  = 0.7    % train set size
    val_sz = 0.2    % validation set size
end
    %% Join the data and labels
    Xy = horzcat(X,y);
    
    %% Shuffle the data and labels
    idx = randperm(size(Xy, 1));
    Xy = Xy(idx, :);
    
    %% Split data by class
    class_names = unique(y);
    M = numel(class_names);
    classes = cell(M,1);
    splits_tr = zeros(M,1);
    for m=1:M
        classes{m} = Xy(y == class_names(m), :);
    end
    
    %% Train set
    for m=1:M
        splits_tr(m) = round(length(classes{m})*tr_sz);
    end
    train = cell(M,1);
    for m=1:M
        train{m} = classes{m}(1:splits_tr(m), :);
    end
    train_set = cell2mat(train);
    train_set = train_set(randperm(length(train_set)),:);
    X_train = train_set(:,1:end-1);
    y_train = train_set(:,end);
    
    %% Validation set
    splits_val = zeros(M,1);
    for m=1:M
        splits_val(m) = splits_tr(m) + round(length(classes{m})*val_sz);
    end
    valid = cell(M,1);
    for m=1:M
        valid{m} = classes{m}(splits_tr(m) + 1:splits_val(m), :);
    end
    valid_set = cell2mat(valid);
    valid_set = valid_set(randperm(length(valid_set)),:);
    X_val = valid_set(:,1:end-1);
    y_val = valid_set(:,end);
    
    %% Test set
    test = cell(M,1);
    for m=1:M
        test{m} = classes{m}(splits_val(m) + 1:end, :);
    end
    test_set = cell2mat(test);
    test_set = test_set(randperm(length(test_set)),:);
    X_test = test_set(:,1:end-1);
    y_test = test_set(:,end);
end

