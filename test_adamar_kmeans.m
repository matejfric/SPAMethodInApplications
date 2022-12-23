%ADAMAR K-MEANS
close all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
rng(42);

ca = load_images();
smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
%descriptors = [Descriptor.Color];
%descriptors = [Descriptor.Roughness];
descriptors = [Descriptor.Roughness Descriptor.Color];
X = get_descriptors(ca, descriptors);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

cluster_counts = 4:1:7;
%cluster_counts = 10:20:100; % Lambda contains values 0.5 only WHY???
%cluster_counts = [4,10,100,1000];
%cluster_counts = size(X,1); % 0.4 precision, i.e. uplně špatně
maxIters = 10;
alpha = 0.8; % Prioritize clustering
%alpha = 0.5;

for k = 1 : length(cluster_counts)
    [Lambda, C, K, a, b, stats_train] = adamar_kmeans_v2(X, cluster_counts(k), alpha, maxIters);
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;

    disp("Lambda:")
    disp(Lambda) % Transition matrix
    
    img = 68;
    [testingErrors(k), count, stats_test] = adamar_predict(Lambda, C', K, a, b, img);
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

score_plot('Adamar K-means', cluster_counts, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished succesfully.\n");

