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
SCALING = false;
VISUALIZE = false;

descriptors = [Descriptor.Roughness Descriptor.Color ];
testing_images = [68, 137, 143];

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

if strcmp(DATASET, 'Dataset2')
    %ca = matrix2ca('Dataset2/Descriptors/');
    ca = matrix2ca('Dataset2/Descriptors512GLRLM/');
    %ca = matrix2ca('Dataset2/DescriptorsProbability/');
    n = numel(ca);
    n_train = floor(n * 0.8);
    n_test = n - n_train;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    %ca_Y = ca(1:n_train); % test on training data
    ca_Y = ca(n_train+1:n); % test on testing data
else
    % X = get_descriptors(load_images();, descriptors);
    X = matfile('X10.mat').X;
    
    n = numel(testing_images);
    ca_Y = cell(n,1);
    for i=1:n
        Y.X = get_descriptors(load_images(testing_images(i)), descriptors); 
        Y.I = testing_images(i);
        ca_Y{i} = Y;
    end
    [X, ca_Y] = scaling(X, ca_Y, 'none');
end

% Removal of strongly correlated columns
[X, ca_Y] = correlation_analysis(X, ca_Y);

% Scaling
%[X, ca_Y] = scaling(X, ca_Y, 'minmax');

% Perform PCA
%[X, ca_Y] = principal_component_analysis(X, ca_Y);

folder = 'Dataset/SmallImagesDescriptors/';
%ca_Y = matrix2ca(folder);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%K = 2:16;
%K = [2,5,12];
%K = 10;
%K = 100;

K = 10;%10;
maxIters = 15;

%alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5];
%alpha = 1e-3.*[1:2:10];
%alpha = 1e-4.*[1:2:10];
alpha = 0:0.1:1;

L1s = zeros(numel(alpha),length(K));
L2s = zeros(numel(alpha),length(K));

PiY = [X(:,end),1-X(:,end)];

for a = 1:numel(alpha)
    for k = 1 : length(K)
        [Lambda, C, Gamma, stats_train, L_out, PiX] = adamar_kmeans(X(:,1:end-1), PiY', K(k), alpha(a), maxIters);
        lprecision(a,k) = stats_train.precision;
        lrecall(a,k) = stats_train.recall;
        lf1score(a,k) = stats_train.f1score;
        laccuracy(a,k) = stats_train.accuracy;

        L1s(a,k) = L_out.L1;
        L2s(a,k) = L_out.L2;

        disp("Lambda:"); disp(Lambda); % Transition matrix
        %    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
        images = 68;
        
        [stats_test] = adamar_predict_mat(Lambda, C', K(k), alpha(a), [], [], ca_Y, DATASET, VISUALIZE);
        %if ~SCALING; [stats_test] = adamar_predict(Lambda, C', K, alpha(a), [], [], images, descriptors); end
        %if SCALING; [stats_test] = adamar_predict(Lambda, C', K, alpha(a), colmin, colmax, images, descriptors); end
        %if ~SCALING;[stats_test] = adamar_predict_mat(Lambda, C', K, alpha(a), [], [], ca_Y, DATASET, false); end
        %if SCALING; [stats_test] = adamar_predict_mat(Lambda, C', K, alpha(a), colmin, colmax, ca_Y, DATASET, false); end
        tprecision(a,k) = stats_test.precision;
        trecall(a,k) = stats_test.recall;
        tf1score(a,k) = stats_test.f1score;
        taccuracy(a,k) = stats_test.accuracy;
    end

    score_plot(sprintf('Adamar k-means, K=%d, alpha=%.2e', K, alpha(a)), K, lprecision(a,:), lrecall(a,:), lf1score(a,:), laccuracy(a,:), tprecision(a,:), trecall(a,:), tf1score(a,:), taccuracy(a,:))

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
    plot(L1s(:,k), L2s(:,k),'r-o');
    %text(L1s(1),L2s(1),['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
    %text(L1s(end),L2s(end),['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
    for i = 1:numel(L1s(:,k))
        text(L1s(i),L2s(i),['$\alpha = ' num2str(alpha(i)) '$'],'Interpreter','latex')
    end
    hold off
end
