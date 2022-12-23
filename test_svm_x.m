close all
clear all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM')

rng(42);

ca = load_images();
%descriptors = [Descriptor.Color];
%descriptors = [Descriptor.Roughness];
%descriptors = [Descriptor.RoughnessGLRL];
descriptors = [Descriptor.Roughness Descriptor.Color];
%descriptors = [Descriptor.Roughness Descriptor.Color Descriptor.RoughnessGLRL];
%descriptors = [Descriptor.Roughness Descriptor.RoughnessGLRL];
X = get_descriptors(ca, descriptors);
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

% TEST SVM ON RAW DATA

range = 1:3; % Numbers of runs

%testing_images = 6:10;
testing_images = 68;

for i = progress(range)
    [stats_train, stats_test] = svm_x(X, testing_images, descriptors);
    
    %C = cell2mat(struct2cell(stats_learn)); % Convert struct to vector
    
    lprecision(i) = stats_train.precision;
    lrecall(i) = stats_train.recall;
    lf1score(i) = stats_train.f1score;
    laccuracy(i) = stats_train.accuracy;
    
    tprecision(i) = stats_test.precision;
    trecall(i) = stats_test.recall;
    tf1score(i) = stats_test.f1score;
    taccuracy(i) = stats_test.accuracy;
end

% PLOTS

% Training
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
xlabel('Run')
ylabel('Score')
legend('Precision','Recall', 'F1-score', 'Accuracy')
title('Training Phase')
hold off

% Testing
subplot(1,2,2)
set(gca,'DefaultLineLineWidth',2)
plot(range,tprecision) 
hold on 
plot(range,trecall)
hold on 
plot(range,tf1score)
hold on 
plot(range,taccuracy)
xlabel('Run')
ylabel('Score')
legend('Precision','Recall', 'F1-score', 'Accuracy')
title('Testing Phase')
sgtitle('SVM X')
hold off


fprintf("\nProgram finished succesfully.\n");

 