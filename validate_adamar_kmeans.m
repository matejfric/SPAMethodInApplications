%ADAMAR K-MEANS
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
rng(42);

descriptors = [Descriptor.Roughness Descriptor.Color ];

ca = matrix2ca('Dataset2/Descriptors/');
n = numel(ca);
n_train = floor(n * 0.7);
n_test = n - n_train;


X = cell2mat({cell2mat(ca(1:n_train)).X}'); % this monstrosity does the same thing as 4 lines below
% X = [];
% for i = 1:n_train
%     X = [X; ca{i}.X];
% end

ca_Y = ca(n_train+1:n);

% % MinMaxScaling [0,1], maybe try also [-1,1]
% colmin = min(X); % a
% colmax = max(X); % b
% X = rescale(X,'InputMin',colmin,'InputMax',colmax);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%cluster_counts = 2:16;
cluster_counts = 4:2:12;
%cluster_counts = 10:5:50;
%cluster_counts = 10:20:100; % Lambda contains values 0.5 only, only 1 state is present
%cluster_counts = [100,1000]; % Lambda contains values 0.5 only, only 1 state is present
%cluster_counts = size(X,1); % Incorrect result
maxIters = 1000;
%alpha = 0.8; % Prioritize clustering
alpha = 0.99;

%alpha = 0.1:0.1:1;
%for a = 1:numel(alpha)

for k = 1 : length(cluster_counts)
    [Lambda, C, K, stats_train] = adamar_kmeans(X, cluster_counts(k), alpha, maxIters);
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;

    disp("Lambda:")
    disp(Lambda) % Transition matrix
    
    %images = 68;
    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
    images = smaller_images;
    
    %[stats_test] = adamar_predict(Lambda, C', K, colmin, colmax, images, descriptors);
    [stats_test] = adamar_predict_mat(Lambda, C', K, [], [], ca_Y);
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

score_plot('Adamar K-means', cluster_counts, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

%end

fprintf("\nProgram finished successfully.\n");

