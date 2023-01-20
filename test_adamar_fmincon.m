%ADAMAR FMINCON()
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon')
addpath('ProgramFiles/SPG')
rng(42);
DATASET = 'Dataset';

small_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
descriptors = [Descriptor.Roughness Descriptor.Color];

%X = matfile('X10.mat').X; % descriptors for first 10 images
X = cell2mat({cell2mat(matrix2ca('Dataset/SmallImagesDescriptors/')).X}'); % descriptors for "smaller" images
%X = get_descriptors(load_images(137), descriptors); % hřebík

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%ADAMAR
%alpha = 1e-4;
%alpha = [1e-4, 1e-3];
alpha = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1-1e-1, 1-1e-2, 1-1e-3];
K = 5; % Number of clusters
maxIters = 2;

for a = 1:numel(alpha)
[C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats_train, L] = ...
    adamar_fmincon(X, K, alpha(a), maxIters);
disp(Lambda);
Ls(a)=L;
end

%EVALUATION
for i = 1:maxIters
    lprecision(i) = stats_train(i).precision;
    lrecall(i) = stats_train(i).recall;
    lf1score(i) = stats_train(i).f1score;
    laccuracy(i) = stats_train(i).accuracy;
end
     
images = small_images;
%images = 137;
for i= 1:numel(images)
    [stats_test] = adamar_predict(Lambda, C, K, [], [], images(i), descriptors);
    
    tprecision(i) = stats_test.precision;
    trecall(i) = stats_test.recall;
    tf1score(i) = stats_test.f1score;
    taccuracy(i) = stats_test.accuracy;
end

%PLOTS
%fmincon_score_plot('Adamar fmincon', 1:maxIters, 1:numel(images),lprecision, lrecall, lf1score, laccuracy, tprecision, trecall, tf1score, taccuracy)   

for k=1:length(K)
    figure
    hold on
    title(['K = ' num2str(K)])
    plot([Ls.L1], [Ls.L2],'r.-');
    hold off
end


