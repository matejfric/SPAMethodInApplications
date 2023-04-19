function [X_train, X_test, y_train, y_test] = kfold_crossval(X, y, k)
%KFOLD_CROSSVAL Prepare data for k-fold cross validation
arguments
    X (:,:) double
    y double
    k = 5 
end
    % Allocate the folds
    Xy_train = cell(k,1);
    Xy_test = cell(k,1);
    
    % Split percentage
    split = 1 / k;

    % Join the data and labels
    Xy = horzcat(X,y);
    
    % Shuffle the data and labels
    idx = randperm(size(Xy, 1));
    Xy = Xy(idx, :);
    
    % Split data by class
    class_names = unique(y);
    M = numel(class_names);
    classes = cell(M,1);
    folds = cell(M,1);
    splits = cell(M,1);
    for m = 1:M
        classes{m} = Xy(y == class_names(m), :);
        folds{m}=cell(k,1);
        splits{m}=floor(length(classes{m})*split);
    end
    
    % Split the data into k folds
    for kk=1:k
        if kk==k
            for m=1:M
                folds{m}{kk} = classes{m}(1+(kk-1)*splits{m}:end, :);
            end
        end
        for m=1:M
            folds{m}{kk} = classes{m}(1+(kk-1)*splits{m}:kk*splits{m}, :);
        end
    end
    
    % Prepare train test splits
    for kk=1:k
        tmp_test = cell(M,1);
        for m=1:M
            tmp_test{m} = vertcat(folds{m}{kk});
        end
        tmp_test = cell2mat(tmp_test);
        tmp_test = tmp_test(randperm(length(tmp_test)),:);
        Xy_test{kk} = tmp_test;
        
        tmp_train_cls = cell(M,1);
        for m=1:M
            tmp_train_folds = cell(k,1);
            for kkk=1:k
                if kkk~=kk
                    tmp_train_folds{kkk} = folds{m}{kkk};
                end
            end
            tmp_train_cls{m} = cell2mat(tmp_train_folds);
        end
        tmp_train_final = cell2mat(tmp_train_cls);
        tmp_train_final = tmp_train_final(randperm(length(tmp_train_final)),:);
        Xy_train{kk} = tmp_train_final;
    end
    
    % Split data and labels
    X_train = cell(k,1);
    X_test = cell(k,1);
    y_train = cell(k,1);
    y_test = cell(k,1);
    for kk=1:k
        X_train{kk} = Xy_train{kk}(:,1:end-1);
        y_train{kk} = Xy_train{kk}(:,end);
        X_test{kk} = Xy_test{kk}(:,1:end-1);
        y_test{kk} = Xy_test{kk}(:,end);
        %fprintf("%d,%d,%d\n", sum(y_train{kk}==1),sum(y_train{kk}==2),sum(y_train{kk}==3))
    end
    
end

