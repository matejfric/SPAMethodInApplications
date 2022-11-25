close all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar

DATASET = 'Dataset';

if false
    number_of_images = 10;
    % Load the dataset
    ca_dataset = load_dataset(number_of_images);
    % Get matrix of graycoprops descriptors with annotations
    fprintf("Performing roughness analysis...\n");
    X = roughness_analysis(ca_dataset);
    fprintf("Roughness analysis finished succesfully.\n");
    %ca_patches = patchify(ca_dataset{1,1}); % Uncomment and debug to see what's going on 
end

if true
    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ]; 
    ca = cell(length(smaller_images),2);
    for i = 1 : length(smaller_images)
        ca{i,1} = imread(sprintf('%s/Original/%d.jpg', DATASET, smaller_images(i)));
        ca{i,2} = imread(sprintf('%s/Annotations/%d.png', DATASET, smaller_images(i)));
    end
    fprintf("Performing roughness analysis...\n");
    X = roughness_analysis(ca);
end
    
fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ",...
    sum(X(:,5)), size(X(:,5), 1)-sum(X(:,5)));

%% Testing

cluster_counts = 2:2:10;
%cluster_counts = 2.^(1:10);

% SVM
if true
    
    class_errX = zeros(1,length(cluster_counts));
    class_errK = zeros(1,length(cluster_counts));
    for k = 1:length(cluster_counts)
        [class_errX(k), class_errK(k)] = kmeans_svm(X, cluster_counts(k));
    end

    figure
    subplot(1,2,1)
    plot(cluster_counts,class_errX)
    title('SVM X Testing Error')
    xlabel('K')
    ylabel('Error')
    
    subplot(1,2,2)
    plot(cluster_counts,class_errK)
    title('SVM Gamma Testing Error')
    xlabel('K')
    ylabel('Error')
end
fprintf("\n")

%KL+Jensen
if true
    class_errX = zeros(1,length(cluster_counts));
    class_errY = zeros(1,length(cluster_counts));
    for k = 1:length(cluster_counts)
        [class_errY(k), class_errX(k)] = kmeans_kl(X,cluster_counts(k));
    end

    figure
    subplot(1,2,1)
    plot(cluster_counts,class_errX)
    title('KL+Jensen Training Error')
    xlabel('K')
    ylabel('Error')

    subplot(1,2,2)
    plot(cluster_counts,class_errY)
    title('KL+Jensen Testing Error')
    xlabel('K')
    ylabel('Error')
end

fprintf("\nProgram finished succesfully.");


 