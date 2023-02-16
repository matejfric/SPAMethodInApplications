% TEST SVM ON RAW DATA
close all
clear all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM')

rng(42);

COLOR = true;
PROBS = false;

%descriptors = [Descriptor.Color]; % up to 0.71 f-1score
%descriptors = [Descriptor.Roughness]; % up to 0.5 f-1score
descriptors = [Descriptor.Roughness Descriptor.Color];
%descriptors = [Descriptor.Roughness Descriptor.Color Descriptor.RoughnessGLRL];
dataset = 'Dataset';
testing_images = [68, 137, 143];

if strcmp(dataset, 'Dataset2')
    %ca = matrix2ca('Dataset2/Descriptors/');
    ca = matrix2ca('Dataset2/Descriptors512GLRLM/');
    %ca = matrix2ca('Dataset2/DescriptorsProbability/');
    n = numel(ca);
    n_train = floor(n * 0.8);
    n_test = n - n_train;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    ca_Y = ca(n_train+1:n);
    [X, ca_Y] = scaling(X, ca_Y, 'none');
    %[X, ca_Y] = correlation_analysis(X, ca_Y); %~0.21
    %[X, ca_Y] = principal_component_analysis(X, ca_Y); %~0.03
else
    X = get_descriptors(load_images(), descriptors, COLOR, PROBS);
    %X = matfile('X10.mat').X;
    
    n = numel(testing_images);
    ca_Y = cell(n,1);
    for i=1:n
        Y.X = get_descriptors(load_images(testing_images(i)), descriptors, COLOR, PROBS); 
        Y.I = testing_images(i);
        ca_Y{i} = Y;
    end
    
    [X, ca_Y] = correlation_analysis(X, ca_Y);
    %[X, ca_Y] = principal_component_analysis(X, ca_Y); %~0.03
    [X, ca_Y] = scaling(X, ca_Y, 'minmax');
end
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));


numbers_of_runs = 1; % Numbers of runs
for i = 1:numbers_of_runs
    [stats_train, stats_test] = svm_x(X, testing_images, descriptors, ca_Y, dataset);
    
    lprecision(i) = stats_train.precision;
    lrecall(i) = stats_train.recall;
    lf1score(i) = stats_train.f1score;
    laccuracy(i) = stats_train.accuracy;
    
    tprecision(i) = stats_test.precision;
    trecall(i) = stats_test.recall;
    tf1score(i) = stats_test.f1score;
    taccuracy(i) = stats_test.accuracy;
end

score_plot('SVM X', 1:numbers_of_runs, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished succesfully.\n");



 