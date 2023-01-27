% TEST SVM ON RAW DATA
close all
clear all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM')

rng(42);

descriptors = [Descriptor.Roughness Descriptor.Color];

if true %DATASET2
    ca = matrix2ca('Dataset2/Descriptors/');
    %ca = matrix2ca('Dataset2/Descriptors512GLRLM/');
    %ca = matrix2ca('Dataset2/DescriptorsProbability/');
    n = numel(ca);
    n_train = floor(n * 0.8);
    n_test = n - n_train;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    ca_Y = ca(n_train+1:n);
else
    ca = load_images();
    X = get_descriptors(ca, descriptors);
end
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

numbers_of_runs = 1; % Numbers of runs

%testing_images = 6:10;
testing_images = 68;

for i = 1:numbers_of_runs
    [stats_train, stats_test] = svm_x(X, testing_images, descriptors, ca_Y);
    
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

 