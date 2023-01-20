%ADAMAR K-MEANS
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans

rng(42);

SCALING = false;

descriptors = [Descriptor.Roughness Descriptor.Color ];

ca = matrix2ca('Dataset2/Descriptors/');
%ca = matrix2ca('Dataset2/Descriptors512GLRLM/');
%ca = matrix2ca('Dataset2/DescriptorsProbability/');
n = numel(ca);
n_train = floor(n * 0.8);
n_test = n - n_train;
X = cell2mat({cell2mat(ca(1:n_train)).X}');
ca_Y = ca(n_train+1:n);

if SCALING
    % % MinMaxScaling [0,1]
    colmin = min(X); % a
    colmax = max(X); % b
    % X = rescale(X,'InputMin',colmin,'InputMax',colmax);

    % Selective MinMaxScaling [-1,1]
    u = 1;
    l = -1;
    cols = colmax > 1; % Select columns to be scaled
    X(:,cols) = l + ...
        ((X(:,cols)-colmin(cols))./(colmax(cols)-colmin(cols))).*(u-l);
end

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%cluster_counts = 2:16;
cluster_counts = 4:2:12;
%cluster_counts = 10:5:50;
maxIters = 1000;
%alpha = 0.5;
alpha = 1e-4;

for k = 1 : length(cluster_counts)
    [Lambda, C, Gamma, K, stats_train, L_out] = adamar_kmeans(X, cluster_counts(k), alpha, maxIters);
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;

    disp("Lambda:")
    disp(Lambda) % Transition matrix
    
    
%     smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
%     images = smaller_images;
    images = 68;
    
    %[stats_test] = adamar_predict(Lambda, C', K, [], [], images, descriptors);
    %[stats_test] = adamar_predict(Lambda, C', K, colmin, colmax, images, descriptors);
    [stats_test] = adamar_predict_mat(Lambda, C', K, [], [], ca_Y);
    %[stats_test] = adamar_predict_mat(Lambda, C', K, colmin, colmax, ca_Y);
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

score_plot('Adamar K-means', cluster_counts, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished successfully.\n");

