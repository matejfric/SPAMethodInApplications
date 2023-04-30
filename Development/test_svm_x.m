% TEST SVM ON RAW DATA
close all
clear all
addpath(genpath(pwd));

rng(42);

dataset = 'Dataset';
testing_images = [68, 137, 143];

[X, ca_Y] = get_train_test_data(dataset);

% Removal of strongly correlated columns
[X, ca_Y] = correlation_analysis(X, ca_Y);

% Scaling
%[X, ca_Y] = scaling(X(:,1:end-1), ca_Y, 'minmax');

% PCA
%[X, ca_Y] = principal_component_analysis(X, ca_Y);
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));


numbers_of_runs = 1; % Numbers of runs
for i = 1:numbers_of_runs
    [stats_train, stats_test] = svm_x(X, testing_images, [], ca_Y, dataset);
    
    lprecision(i) = stats_train.precision;
    lrecall(i) = stats_train.recall;
    lf1score(i) = stats_train.f1score;
    laccuracy(i) = stats_train.accuracy;
    
    tprecision(i) = stats_test.precision;
    trecall(i) = stats_test.recall;
    tf1score(i) = stats_test.f1score;
    taccuracy(i) = stats_test.accuracy;
end

score_plot('SVM X', 1:numbers_of_runs, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished succesfully.\n");



 