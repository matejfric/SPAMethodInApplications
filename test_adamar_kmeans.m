close all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans

DATASET = 'Dataset';

if false
    number_of_images = 10;
    
    % Load the dataset
    ca_dataset = load_dataset(number_of_images);

    % Get matrix of graycoprops descriptors with annotations
    X = roughness_analysis(ca_dataset);
    %ca_patches = patchify(ca_dataset{1,1}); % Uncomment to see what's going on. 
end

if true
    % Load some (smaller) images:
    % Selection of images bellow gives fairly balanced training dataset 
    % (Ones: 1138.00, Zeros: 2458.00). 
    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ]; 
    ca = cell(length(smaller_images),2);
    for i = 1 : length(smaller_images)
        ca{i,1} = imread(sprintf('%s/Original/%d.jpg', DATASET, smaller_images(i)));
        ca{i,2} = imread(sprintf('%s/Annotations/%d.png', DATASET, smaller_images(i)));
    end
    X = roughness_analysis(ca);
end

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,5)), size(X(:,5), 1)-sum(X(:,5)));

cluster_counts = 2:2:10;
%cluster_counts = 10:10:100; % Lambda contains values 0.5 only
%cluster_counts = [4,10,100,1000];
%cluster_counts = size(X,1); % 0.4 precision, i.e. uplně špatně
maxIters = 1000;
%alpha = 0.7; % Prioritize clustering
alpha = 0.5;
learningPrecision = 1:length(cluster_counts);
testingPrecision = 1:length(cluster_counts);
for k = 1 : length(cluster_counts)
    [Lambda, C, K, a, b] = adamar_kmeans(X, cluster_counts(k), alpha, maxIters);
    
    disp("Lambda:")
    disp(Lambda) % Transition matrix
    
    %LEARNING ERRORS
    learningErrors = zeros(0,size(smaller_images,2));
    fprintf("Learning errors: <error (sum(prediction), sum(trueLabels))>\n")
    for i= 1:size(learningErrors,2)
        img = smaller_images(i);
        [learningErrors(i), count] = adamar_predict(Lambda, C', K, a, b, img);
        fprintf("%.2f (%d, %d),\n", learningErrors(i), count(1), count(2));
    end
    learningPrecision(k) = sum(learningErrors) / size(learningErrors,2);
    fprintf("learning precision: %.2f\n", learningPrecision(k));
    
    %TESTING ERRORS
    testingErrors = zeros(0,5);
    fprintf("\nTesting errors: <error (sum(prediction), sum(trueLabels))>\n")
    for i= 1:size(testingErrors,2)
        [testingErrors(i), count] = adamar_predict(Lambda, C', K, a, b, i);
        fprintf("%.2f (%d, %d),\n", testingErrors(i), count(1), count(2));
    end
    testingPrecision(k) = sum(testingErrors) / size(testingErrors,2);
    fprintf("testing precision: %.2f\n", testingPrecision(k));
end

figure
subplot(1,2,1)
plot(cluster_counts, learningPrecision)
title('Training Error')

subplot(1,2,2)
plot(cluster_counts, testingPrecision)
title('Testing Error')

