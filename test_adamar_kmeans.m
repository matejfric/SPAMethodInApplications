%ADAMAR K-MEANS
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
rng(42);

descriptors = [Descriptor.Roughness Descriptor.Color ];

if false % Save matrix X
    ca = load_images(26:40); % Processing of 25 images ~ 410 seconds
    %descriptors = [Descriptor.Color];
    %descriptors = [Descriptor.Roughness];
    %descriptors = [Descriptor.Roughness Descriptor.RoughnessGLRL Descriptor.Color ];
    tic
    X = get_descriptors(ca, descriptors);
    toc
    
    save('X26-40.mat','X');
end

save_X = matfile('X10.mat');
X = save_X.X;

% % Normalization
% X(:,1:end-1) = normalize(X(:,1:end-1));

% % MinMaxScaling [0,1], maybe try also [-1,1]
% colmin = min(X); % a
% colmax = max(X); % b
% X = rescale(X,'InputMin',colmin,'InputMax',colmax);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%cluster_counts = 2:16;
cluster_counts = 4:2:14;
%cluster_counts = 10:20:100; % Lambda contains values 0.5 only, only 1 state is present
%cluster_counts = [4,10,100,1000];
%cluster_counts = size(X,1); % Incorrect result
maxIters = 1000;
%alpha = 0.8; % Prioritize clustering
alpha = 0.5;

%alpha = 0.1:0.1:1;

for a = 1:numel(alpha)

for k = 1 : length(cluster_counts)
    [Lambda, C, K, stats_train] = adamar_kmeans(X, cluster_counts(k), alpha(a), maxIters);
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;

    disp("Lambda:")
    disp(Lambda) % Transition matrix
    
    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
    images = smaller_images;
    images = 68;
    
    %[stats_test] = adamar_predict(Lambda, C', K, colmin, colmax, images, descriptors);
    [stats_test] = adamar_predict(Lambda, C', K, [], [], images, descriptors);
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

score_plot('Adamar K-means', cluster_counts, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

end

fprintf("\nProgram finished successfully.\n");

