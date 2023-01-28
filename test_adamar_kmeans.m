%ADAMAR K-MEANS
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
addpath('ProgramFiles/SPG') 
rng(42);

DATASET = 'Dataset2';

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

if strcmp(DATASET, 'Dataset2')
    ca = matrix2ca('Dataset2/Descriptors/');
    %ca = matrix2ca('Dataset2/Descriptors512GLRLM/');
    %ca = matrix2ca('Dataset2/DescriptorsProbability/');
    n = numel(ca);
    n_train = floor(n * 0.8);
    n_test = n - n_train;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    %ca_Y = ca(1:n_train); % test on training data
    ca_Y = ca(n_train+1:n); % test on testing data
end

folder = 'Dataset/SmallImagesDescriptors/';
%ca_Y = matrix2ca(folder);

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

%K = 2:16;
%K = [2,5,12];
K = 10;

%K = 4:2:14;
maxIters = 1000;

%alpha = [1e-12,1e-8, 1e-4,1e-3,1e-2];
%alpha = 1e-4;
%alpha = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5];
%alpha = [1e-8, 1e-4, 1e-2, 1e-1, 0.5, 1-1e-1, 1-1e-2, 1-1e-4, 1-1e-8];
%alpha = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3];
alpha = 1e-3:2e-3:9e-3; % 0.0010    0.0030    0.0050    0.0070    0.0090

L1s = zeros(numel(alpha),length(K));
L2s = zeros(numel(alpha),length(K));

for a = 1:numel(alpha)
    for k = 1 : length(K)
        [Lambda, C, Gamma, K, stats_train, L_out] = adamar_kmeans(X, K(k), alpha(a), maxIters);
        lprecision(a,k) = stats_train.precision;
        lrecall(a,k) = stats_train.recall;
        lf1score(a,k) = stats_train.f1score;
        laccuracy(a,k) = stats_train.accuracy;

        L1s(a,k) = L_out.L1;
        L2s(a,k) = L_out.L2;

        %   disp("Lambda:")
        %   disp(Lambda) % Transition matrix
        %    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
        images = 68;

        %[stats_test] = adamar_predict(Lambda, C', K, alpha(a), colmin, colmax, images, descriptors);
        %[stats_test] = adamar_predict(Lambda, C', K, alpha(a), [], [], images, descriptors);
        [stats_test] = adamar_predict_mat(Lambda, C', K, alpha(a), [], [], ca_Y, DATASET);
        %[stats_test] = adamar_predict_mat(Lambda, C', K, alpha(a), colmin, colmax, ca_Y, DATASET);
        tprecision(a,k) = stats_test.precision;
        trecall(a,k) = stats_test.recall;
        tf1score(a,k) = stats_test.f1score;
        taccuracy(a,k) = stats_test.accuracy;
    end

    score_plot('Adamar K-means', K, ...
        lprecision(a,:), lrecall(a,:), lf1score(a,:), laccuracy(a,:),...
        tprecision(a,:), trecall(a,:), tf1score(a,:), taccuracy(a,:))

end

% regularization_plot(sprintf('Adamar K-means, k=%d', K), alpha, ...
%     lprecision, lrecall, lf1score, laccuracy,...
%     tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished successfully.\n");

% L-curve
for k=1:length(K)
    figure
    hold on
    title(sprintf('akmeans, K=%d', K))
    plot(L1s(:,k), L2s(:,k),'r.-');
    %text(L1s(1),L2s(1),['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
    %text(L1s(end),L2s(end),['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
    for i = 1:numel(L1s(:,k))
        text(L1s(i),L2s(i),['$\alpha = ' num2str(alpha(i)) '$'],'Interpreter','latex')
    end
    hold off
end
