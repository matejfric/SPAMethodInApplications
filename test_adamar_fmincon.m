close all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon')
rng(42);
DATASET = 'Dataset';

if false
    % Results in allocation error (100 GB RAM) => rescale images OR find a better algorithm
    number_of_images = 10;
    
    % Load the dataset
    ca_dataset = load_dataset(number_of_images);

    % Get matrix of graycoprops descriptors with annotations
    X = roughness_analysis(ca_dataset);
    %ca_patches = patchify(ca_dataset{1,1}); % Uncomment to see what's going on. 
end

if true
    % Load some (smaller!) images:
    % Selection of images bellow gives fairly balanced training dataset 
    % (Ones: 1138.00, Zeros: 2458.00). 
    
    % The more images, the lesser F-count must be set in fmincon, otherwise
    % MATLAB will run out of memory. It is impossible to reach low learning
    % error for multiple images.
     
 %   smaller_images = [137];
     smaller_images = [137, 172, 228, 240];
%    smaller_images = [ 172, 177, 179, 203];
%    smaller_images = [ 172, 177, 179, 203, 209, 212, 228, 240 ]; % Balanced
    ca = cell(length(smaller_images),2);
    for i = 1 : length(smaller_images)
        ca{i,1} = imread(sprintf('%s/Original/%d.jpg', DATASET, smaller_images(i)));
        ca{i,2} = imread(sprintf('%s/Annotations/%d.png', DATASET, smaller_images(i)));
    end
    X = [roughness_analysis(ca), color_analysis(ca), get_ground_truth(ca)];
end

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

K = 10; % Number of clusters
adamar_helper(X, K);