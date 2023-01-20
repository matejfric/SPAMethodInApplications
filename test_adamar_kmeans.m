%ADAMAR K-MEANS
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
addpath('ProgramFiles/SPG') 
rng(42);

descriptors = [Descriptor.Roughness Descriptor.Color ];
% ca = load_images();
% X = get_descriptors(ca, descriptors);

if false % Save matrix X
    ca = load_images([ 172, 177, 179, 203, 209, 212, 228, 240 ]); % Processing of 25 images ~ 410 seconds
    %descriptors = [Descriptor.Color];
    %descriptors = [Descriptor.Roughness];
    %descriptors = [Descriptor.Roughness Descriptor.RoughnessGLRL Descriptor.Color ];
    tic
    X = get_descriptors(ca, descriptors);
    toc
    
    save('X__.mat','X');
end

save_X = matfile('X10.mat');
X = save_X.X;

folder = 'Dataset/SmallImagesDescriptors/';
ca_Y = matrix2ca(folder);

% % Normalization
% X(:,1:end-1) = normalize(X(:,1:end-1));

% % MinMaxScaling [0,1], maybe try also [-1,1]
% colmin = min(X); % a
% colmax = max(X); % b
% X = rescale(X,'InputMin',colmin,'InputMax',colmax);

% % Selective MinMaxScaling [-1,1]
% colmin = min(X); % a
% colmax = max(X); % b
% u = 1;
% l = -1;
% cols = colmax > 1; % Select columns to be scaled
% X(:,cols) = l + ...
%     ((X(:,cols)-colmin(cols))./(colmax(cols)-colmin(cols))).*(u-l);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%cluster_counts = 2:16;
cluster_counts = 5;%[2,5,12];%12;
%cluster_counts = 4:2:14;
%cluster_counts = 10:20:100; % Lambda contains values 0.5 only, only 1 state is present
%cluster_counts = [4,10,100,1000];
%cluster_counts = size(X,1); % Incorrect result
maxIters = 1000;
%alpha = 0.8; % Prioritize clustering
%alpha = 0.5;

%alpha = 0.1:0.1:1;
%alpha = [1e-12,1e-8, 1e-4,1e-3,1e-2];
%alpha = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5];
alpha = [1e-8, 1e-4, 1e-2, 1e-1, 0.5, 1-1e-1, 1-1e-2, 1-1e-4, 1-1e-8];
%alpha = 1e-2;

L1s = zeros(numel(alpha),length(cluster_counts));
L2s = zeros(numel(alpha),length(cluster_counts));

for k = 1:numel(alpha)

for a = 1 : length(cluster_counts)
    [Lambda, C, Gamma, K, stats_train, L_out] = adamar_kmeans(X, cluster_counts(a), alpha(k), maxIters);
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;

    L1s(k,a) = L_out.L1;
    L2s(k,a) = L_out.L2;
    
 %   disp("Lambda:")
 %   disp(Lambda) % Transition matrix
    
%    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
%    images = smaller_images;
    images = 68;
    
    %[stats_test] = adamar_predict(Lambda, C', K, colmin, colmax, images, descriptors);
    [stats_test] = adamar_predict(Lambda, C', K, [], [], images, descriptors);
    %[stats_test] = adamar_predict_mat(Lambda, C', K, [], [], ca_Y);
    %[stats_test] = adamar_predict_mat(Lambda, C', K, colmin, colmax, ca_Y);
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

% score_plot('Adamar K-means', cluster_counts, ...
%     lprecision, lrecall, lf1score, laccuracy,...
%     tprecision, trecall, tf1score, taccuracy)

end

regularization_plot(sprintf('Adamar K-means, k=%d', cluster_counts), alpha, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished successfully.\n");

for k=1:length(cluster_counts)
figure
hold on
title(['K = ' num2str(cluster_counts(k))])
plot(L1s(:,k),L2s(:,k),'r.-');
hold off
end
