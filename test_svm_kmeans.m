% TEST SVM WITH K-MEANS AND GAMMA MATRIX
close all
clear all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar

rng(42);

ca = load_images();

%descriptors = [Descriptor.Roughness Descriptor.Color Descriptor.RoughnessGLRL];
descriptors = [];
X = get_descriptors(ca, descriptors);
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%cluster_counts = 2:2:10;
cluster_counts = 2.^(1:8);

% TRAINING
for k = progress(1:length(cluster_counts))
    [stats_train, stats_test] = svm_kmeans(X, cluster_counts(k), 68, descriptors);
    
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;
    
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

score_plot('SVM KMEANS', cluster_counts, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)


fprintf("\nProgram finished succesfully.\n");


 