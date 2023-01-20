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
alpha = 1e-4;
K = 5; % Number of clusters
maxIters = 3;
[C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats_train] = ...
    adamar_fmincon(X, K, alpha, maxIters);

disp(Lambda);

%EVALUATION
for i = 1:maxIters
    lprecision(i) = stats_train(i).precision;
    lrecall(i) = stats_train(i).recall;
    lf1score(i) = stats_train(i).f1score;
    laccuracy(i) = stats_train(i).accuracy;
end
     
images = small_images;
for i= 1:numel(images)
    [stats_test] = adamar_predict(Lambda, C, K, [], [], images(i), descriptors);
    
    tprecision(i) = stats_test.precision;
    trecall(i) = stats_test.recall;
    tf1score(i) = stats_test.f1score;
    taccuracy(i) = stats_test.accuracy;
end

%PLOTS
fmincon_score_plot('Adamar K-means', 1:maxIters, 1:numel(images), ...
        lprecision, lrecall, lf1score, laccuracy,...
        tprecision, trecall, tf1score, taccuracy)

