% TEST SVM ON RAW DATA
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

range = 1:3; % Numbers of runs

%testing_images = 6:10;
testing_images = 68;

for i = progress(range)
    [stats_train, stats_test] = svm_x(X, testing_images, descriptors);
    
    lprecision(i) = stats_train.precision;
    lrecall(i) = stats_train.recall;
    lf1score(i) = stats_train.f1score;
    laccuracy(i) = stats_train.accuracy;
    
    tprecision(i) = stats_test.precision;
    trecall(i) = stats_test.recall;
    tf1score(i) = stats_test.f1score;
    taccuracy(i) = stats_test.accuracy;
end

score_plot('SVM X', range, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished succesfully.\n");

 