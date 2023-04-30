%--------------------------------------------------------------------------    
%  Comparison on Diagnostic Wisconsin Breast Cancer Dataset
%--------------------------------------------------------------------------

clear all
warning off; % Custom warning that MCC is undefined during training. 
addpath(genpath(pwd));
rng(42);

[X,y] = get_bcwd_data();
%[X,y] = get_wine_data();
%[X,y] = get_ionosphere_data();
%X = normalize(X,'range');
X = normalize(X,'zscore');

if false
    % Visualizing the dataset distribution
    opts = detectImportOptions('bcwd.csv','NumHeaderLines',0);
    table = readtable('bcwd.csv',opts);
    features = table.Properties.VariableNames;
    figure
    tiledlayout(5,6)
    for d = 1:size(X,2)
        nexttile
        % Use the pdf option to normalize the area under the histogram to 1
        histogram(X(:,d), 100, 'Normalization', 'pdf')
        title(strrep(sprintf('Distribution of %s', features{d+2}), '_', ' '))
        xlabel('Data Values')
        ylabel('Probability Density')
    end
end

classifiers{1} = KKLDJ();
classifiers{2} = KKLD();
clf_names = ["KKLDJ", "KKLD","KLambda", "KNN", "LinearSVM", "GaussianSVM", "D-Tree"];
%classifiers{3} = SPAKLD();
%clf_names = ["KKLDJ", "KKLD", "SPAKLD", "KLambda", "KNN", "LinearSVM", "GaussianSVM", "D-Tree"];

n_rand = 10; % Number of random runs
for n = progress(1:n_rand)

%--------------------------------------------------------------------------    
%  KLambda & Decision tree & SVM
%--------------------------------------------------------------------------
[X_train, X_test, y_train, y_test] = train_test_split(X,y,0.75);    
    
%KLambda
mdlKL = KLambda();
clf = find(clf_names == "KLambda");
mdlKL.fit(X_train, y_train);
y_pred = mdlKL.predict(X_test');
stats_test{clf}(n,1) = mdlKL.computeStats(y_pred, y_test');

%Decision tree
clf = find(clf_names == "D-Tree");
mdlTree = fitctree(X_train, y_train);
y_pred = mdlTree.predict(X_test);
stats_test{clf}(n,1) = statistics(y_pred, y_test);

%LinearSVM
clf = find(clf_names == "LinearSVM");
mdlSVM = fitcsvm(X_train, y_train,'KernelScale','auto','KernelFunction','linear');
y_pred = mdlSVM.predict(X_test);
stats_test{clf}(n,1) = statistics(y_pred, y_test); 

%GaussianSVM
clf = find(clf_names == "GaussianSVM");
mdlSVM = fitcsvm(X_train, y_train,'KernelScale','auto','KernelFunction','gaussian');
y_pred = mdlSVM.predict(X_test);
stats_test{clf}(n,1) = statistics(y_pred, y_test); 

%--------------------------------------------------------------------------
%  KNN, KKLDJ, KKLD, SPAKLD
%--------------------------------------------------------------------------
[X_train, X_val, X_test, y_train, y_val, y_test] = train_test_val_split(X,y,0.6,0.2);

%KNN
Ks_knn = 1:30;
clf = find(clf_names == "KNN");
for k = Ks_knn
    mdlKNN = fitcknn(X_train, y_train, 'NumNeighbors', k);
    y_pred = mdlKNN.predict(X_val);
    stats_val_knn(k) = statistics(y_pred, y_val);
end
best_mcc_knn = max([stats_val_knn.mcc]);
idx = find([stats_val_knn.mcc] == best_mcc_knn,1,'first');
best_k_knn = Ks_knn(idx);
mdlKNN = fitcknn(X_train, y_train, 'NumNeighbors', best_k_knn);
y_pred = mdlKNN.predict(X_test);
stats_test{clf}(n,1) = statistics(y_pred, y_test);    

%KKLDJ, KKLD, SPAKLD

%Ks = 2:1:5;
Ks = [2,3,4,7,10,15];
alphas = 0:0.1:1;
%alphas = [0.001, 0.01, 0.1, 0.5, 0.8, 0.9, 0.99, 0.999, 0.999, 0.9999];
%alphas = 0.999:0.0002:0.9999;
for clf=1:length(classifiers)
    for a = 1:length(alphas)
        for k = 1:length(Ks)
            classifiers{clf}.alpha = alphas(a);
            classifiers{clf}.K = Ks(k);
            classifiers{clf}.fit(X_train,y_train)
            y_pred = classifiers{clf}.predict(X_test');
            stats_val(a,k) = classifiers{clf}.computeStats(y_pred, y_test');
        end
    end
    [best_alpha, best_K, best_mcc] = grid_search_mcc(stats_val,alphas,Ks);
    classifiers{clf}.alpha = best_alpha;
    classifiers{clf}.K = best_K;
    classifiers{clf}.fit(X_train,y_train);
    y_pred = classifiers{clf}.predict(X_test');
    stats = classifiers{clf}.computeStats(y_pred, y_test');
    stats_test{clf}(n,1) = stats;
end

end

%--------------------------------------------------------------------------
% Compute mean and standard deviation
%--------------------------------------------------------------------------
for clf=1:length(clf_names)
    f1s{clf}      = mean([stats_test{clf}.f1score]);
    acc{clf}      = mean([stats_test{clf}.accuracy]);
    nmcc{clf}     = mean([stats_test{clf}.nmcc]);
    std_f1s{clf}  = std([stats_test{clf}.f1score]);
    std_acc{clf}  = std([stats_test{clf}.accuracy]);
    std_nmcc{clf} = std([stats_test{clf}.nmcc]);
end

%--------------------------------------------------------------------------
% Display results
%--------------------------------------------------------------------------

% Sort nmcc vector in descending order
nmcc_vec = [nmcc{:}];
[nmcc_sorted, idx] = sort(nmcc_vec, 'descend');

% Sort other vector accordingly using the same indexing
clf_names = pad(clf_names(idx),'both');
f1s = f1s(idx);
acc = acc(idx);
nmcc = nmcc(idx);
std_f1s = std_f1s(idx);
std_acc = std_acc(idx);
std_nmcc = std_nmcc(idx);

fprintf("- Testing Scores\n");
for clf=1:length(clf_names)
    fprintf("| %s: nMCC = %.3f ± %.2f  |  F1 = %.3f ± %.2f  |  ACC = %.3f ± %.2f  |\n",...
        clf_names(clf),...
        nmcc{clf},std_nmcc{clf},...
        f1s{clf},std_f1s{clf},...
        acc{clf},std_acc{clf});
end

%--------------------------------------------------------------------------
% Write latex table
%--------------------------------------------------------------------------

% Open the output file for writing
fileID = fopen('comparison_results.txt', 'w');

% Write the testing scores to the output file
fprintf(fileID,"\\toprule\nSML algorithm & NormMCC & Accuracy & $F_1$-score \\\\\n\\midrule\n");
clf_names = pad(clf_names,'both');
for clf=1:length(clf_names)
    fprintf(fileID, "%s & $%.3f \\pm %.2f$  &  $%.3f \\pm %.2f$  &  $%.3f \\pm %.2f$  \\\\\n",...
        clf_names(clf),...
        nmcc{clf},std_nmcc{clf},...    
        f1s{clf},std_f1s{clf},...
        acc{clf},std_acc{clf});
end
fprintf(fileID,"\\bottomrule\n");

% Close the output file
fclose(fileID);
