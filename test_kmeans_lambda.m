% TEST K-MEANS + KL-JENSEN (LAMBDA SOLVER) 
close all
clear all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar

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

%cluster_counts = 2:2:10;
cluster_counts = 2.^(1:8);

for k = progress(1:length(cluster_counts))
    %[stats_train, stats_test] = kmeans_lambda(X, cluster_counts(k), 68, descriptors, []);
    [stats_train, stats_test] = kmeans_lambda(X, cluster_counts(k), 68, descriptors, ca_Y);
    
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;
    
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

score_plot('Lambda Solver (KL+Jensen)', cluster_counts, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)


fprintf("\nProgram finished succesfully.");

