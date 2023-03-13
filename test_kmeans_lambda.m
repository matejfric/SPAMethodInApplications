% TEST K-MEANS + KL-JENSEN (LAMBDA SOLVER) 
close all
clear all
addpath(genpath(pwd));
rng(42);

dataset = 'Dataset256'; % 'Dataset', 'Dataset2', 'Dataset256', 'DatasetSelection'
[X, ca_Y] = get_train_test_data(dataset);

% Removal of strongly correlated columns
%[X, ca_Y] = correlation_analysis(X, ca_Y);

PiY = [X(:,end)'; 1-X(:,end)']; % Ground truth
X = X(:,1:end-1);

% Scaling
[X, ca_Y] = scaling(X, ca_Y, 'minmax');
%[X, ca_Y] = scaling(X, ca_Y, 'zscore', 'robust');

% PCA
%[X, ca_Y, explained] = mypca(X, ca_Y);
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ", sum(PiY(1,:)), sum(PiY(2,:)));

%Ks = 2:2:10;
%Ks = [2.^(1:find_max_K(X)), size(X,1)];
Ks = 2.^(1:8);
for k = progress(1:length(Ks))
    
    K = Ks(k);
    [C, Gamma, PiX, Lambda, ~, stats_train, ~] = kmeans_lambda(X, K, PiY);
    
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;
    
    display = false;
    [stats_test] = adamar_predict(Lambda, C', K, NaN, ca_Y, dataset, display);
    
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

score_plot('Lambda Solver (KL+Jensen)', Ks, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished succesfully.");


function [max_K] = find_max_K(X)
i=1;
x=2^(i);
while x < size(X,1)
     i=i+1;
     x=2^(i);
end
max_K = i-1;
end

