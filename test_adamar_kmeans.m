close all


addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans

DATASET = 'Dataset';

ca = load_images();
smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
%descriptors = [Descriptor.Color];
%descriptors = [Descriptor.Roughness];
descriptors = [Descriptor.Roughness Descriptor.Color];
X = get_descriptors(ca, descriptors);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

cluster_counts = 4:2:10;
%cluster_counts = 10:20:100; % Lambda contains values 0.5 only WHY???
%cluster_counts = [4,10,100,1000];
%cluster_counts = size(X,1); % 0.4 precision, i.e. uplně špatně
maxIters = 10;
alpha = 0.8; % Prioritize clustering
%alpha = 0.5;
learningPrecision = 1:length(cluster_counts);
testingPrecision = 1:length(cluster_counts);
for k = 1 : length(cluster_counts)
    [Lambda, C, K, a, b] = adamar_kmeans_v2(X, cluster_counts(k), alpha, maxIters);

    disp("Lambda:")
    disp(Lambda) % Transition matrix

    %LEARNING ERRORS
    learningErrors = zeros(0,size(smaller_images,2));
    fprintf("Learning errors: <error (sum(prediction), sum(trueLabels))>\n")
    for i= 1:size(learningErrors,2)
        img = smaller_images(i);
        [learningErrors(i), count, stats_train(i)] = adamar_predict(Lambda, C', K, a, b, img);
        lprecision(i) = stats_train(i).precision;
        lrecall(i) = stats_train(i).recall;
        lf1score(i) = stats_train(i).f1score;
        laccuracy(i) = stats_train(i).accuracy;
        fprintf("%.2f (%d, %d),\n", learningErrors(i), count(1), count(2));
    end
    learningPrecision(k) = sum(learningErrors) / size(learningErrors,2);
    fprintf("learning precision: %.2f\n", learningPrecision(k));

    %TESTING ERRORS
    countTests = 5;
    testingErrors = zeros(0,countTests);
    fprintf("\nTesting errors: <error (sum(prediction), sum(trueLabels))>\n")
    for i= 1:size(testingErrors,2)
        [testingErrors(i), count, stats_test(i)] = adamar_predict(Lambda, C', K, a, b, i);
        tprecision(i) = stats_test.precision;
        trecall(i) = stats_test.recall;
        tf1score(i) = stats_test.f1score;
        taccuracy(i) = stats_test.accuracy;
        fprintf("%.2f (%d, %d),\n", testingErrors(i), count(1), count(2));
    end
    testingPrecision(k) = sum(testingErrors) / size(testingErrors,2);
    fprintf("testing precision: %.2f\n", testingPrecision(k));
end

% Error plots:
figure
subplot(1,2,1)
plot(cluster_counts, learningPrecision)
title('Training Error')

subplot(1,2,2)
plot(cluster_counts, testingPrecision)
title('Testing Error')

% Score Plots:

% Training
range = 1:size(learningErrors,2);

figure
subplot(1,2,1)
set(gca,'DefaultLineLineWidth',2)
plot(range,lprecision) 
hold on 
plot(range,lrecall)
hold on 
plot(range,lf1score)
hold on 
plot(range,laccuracy)
xlabel('K')
ylabel('Score')
legend('Precision','Recall', 'F1-score', 'Accuracy')
title('Training Phase')
hold off

% Testing
range = 1:size(testingErrors,2);

subplot(1,2,2)
set(gca,'DefaultLineLineWidth',2)
plot(range,tprecision) 
hold on 
plot(range,trecall)
hold on 
plot(range,tf1score)
hold on 
plot(range,taccuracy)
xlabel('Image')
ylabel('Score')
legend('Precision','Recall', 'F1-score', 'Accuracy')
title('Testing Phase')
sgtitle('Adamar K-means')
hold off

fprintf("\nProgram finished succesfully.\n");

