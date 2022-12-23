close all
clear all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar

rng(42);

ca = load_images();
%descriptors = [Descriptor.Roughness Descriptor.Color Descriptor.RoughnessGLRL];
descriptors = [];
X = get_descriptors(ca, descriptors);
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

% TEST K-MEANS + KL-JENSEN LAMBDA SOLVER 

%cluster_counts = 2:2:10;
cluster_counts = 2.^(1:8);

%lprecision = zeros(1,length(cluster_counts)); %...preallocation

for k = progress(1:length(cluster_counts))
    [stats_train, stats_test] = kmeans_lambda(X, cluster_counts(k), 68, descriptors);
    
    lprecision(k) = stats_train.precision;
    lrecall(k) = stats_train.recall;
    lf1score(k) = stats_train.f1score;
    laccuracy(k) = stats_train.accuracy;
    
    tprecision(k) = stats_test.precision;
    trecall(k) = stats_test.recall;
    tf1score(k) = stats_test.f1score;
    taccuracy(k) = stats_test.accuracy;
end

% PLOTS

% Training
figure
subplot(1,2,1)
set(gca,'DefaultLineLineWidth',2)
plot(cluster_counts,lprecision) 
hold on 
plot(cluster_counts,lrecall)
hold on 
plot(cluster_counts,lf1score)
hold on 
plot(cluster_counts,laccuracy)
xlabel('Number of clusters (K)')
ylabel('Score')
xticks(cluster_counts)
if max(cluster_counts) >= 100
    set(gca, 'XScale', 'log')
end
legend('Precision','Recall', 'F1-score', 'Accuracy',...
    'Location','southeast')
title('KL+Jensen Training Phase')
hold off

% Testing
subplot(1,2,2)
set(gca,'DefaultLineLineWidth',2)
plot(cluster_counts,tprecision) 
hold on 
plot(cluster_counts,trecall)
hold on 
plot(cluster_counts,tf1score)
hold on 
plot(cluster_counts,taccuracy)
xlabel('Number of clusters (K)')
ylabel('Score')
xticks(cluster_counts)
if max(cluster_counts) >= 100
    set(gca, 'XScale', 'log')
end
legend('Precision','Recall', 'F1-score', 'Accuracy',...
    'Location','southeast')
title('Testing Phase')
sgtitle('Lambda Solver (KL+Jensen)')
hold off


fprintf("\nProgram finished succesfully.");

