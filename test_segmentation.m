close all
clear all
addpath(genpath(pwd));

rng(42);

DATASET = 'Segmentation';

img = 68;
input = load_images(img);
X_IN = input{1};
%imshow(X_IN)

fprintf("LBP analysis...\n");
[X_LBP, ~] = get_train_test_data(DATASET, [Descriptor.LBP]);
fprintf("GLCM analysis...\n");
[X_GLCM, ~] = get_train_test_data(DATASET, [Descriptor.Roughness]);
fprintf("Statistical Moments analysis...\n");
[X_SM, ~] = get_train_test_data(DATASET, [Descriptor.Color]);
fprintf("GLRLM analysis...\n");
[X_GLRL, ~] = get_train_test_data(DATASET, [Descriptor.RoughnessGLRL]);

X_LBP  = X_LBP (:,1:end-1);
X_GLCM = X_GLCM(:,1:end-1);
X_SM   = X_SM  (:,1:end-1);
X_GLRL = X_GLRL(:,1:end-1);

K = 5;
fprintf("K-means K=%d...\n", K);
X_LBP_Seg  = kmeans_segmentation(X_LBP , X_IN, K);
X_GLCM_Seg = kmeans_segmentation(X_GLCM, X_IN, K);
X_SM_Seg   = kmeans_segmentation(X_SM  , X_IN, K);
X_GLRL_Seg = kmeans_segmentation(X_GLRL, X_IN, K);

plot_segmentation(X_IN, X_LBP_Seg,X_GLCM_Seg, X_SM_Seg, X_GLRL_Seg)

function [segmentation_image] = kmeans_segmentation(X, X_IN, K)
    arguments
        X, X_IN, K = 4;
    end
    [X, ~] = scaling(X, [], "minmax");
    [cluster_idx, ~, ~] = kmeans(X,K);
    segmentation_image = vector2image(cluster_idx, X_IN);
end


function plot_segmentation(X_IN, X_LBP, X_GLCM, X_SM, X_GLRL)
    fontsize = 12;
    rows = 2;
    cols = 3;
    
    fig = figure();
    fig.WindowState = 'maximized';

    subplot(rows, cols, 1);
    imshow(X_IN);
    title('Input Image', 'FontSize', fontsize, 'Color', 'r');

    subplot(rows, cols, 2);
    imshow(X_LBP, []);
    title('Segmentation LBP', 'FontSize', fontsize, 'Color', 'r');
    
    subplot(rows, cols, 3);
    imshow(X_GLCM, []);
    title('Segmentation GLCM', 'FontSize', fontsize, 'Color', 'r');
    
    subplot(rows, cols, 4);
    imshow(X_SM, []);
    title('Segmentation Stat. Moments', 'FontSize', fontsize, 'Color', 'r');
    
    subplot(rows, cols, 5);
    imshow(X_GLRL, []);
    title('Segmentation GLRLM', 'FontSize', fontsize, 'Color', 'r');
end


function img = vector2image(vector, original_image)
    I = im2double(rgb2gray(original_image)); 
    [ca_patches, ~] = patchify(I);
    [patch_size, ~] = size(cell2mat(ca_patches(1,1)));
    [rows, cols] = size(ca_patches);
    
    vector = normalize(vector, 'range');
    
    for i=1:rows
        for j=1:cols
           ca_patches{i,j} = zeros(patch_size) + vector((i-1)*cols+j);
        end
    end
    
    img = cell2mat(ca_patches);
end


function sse_plot(X, Ks)
    arguments
        X {double}
        Ks = 2:2:10;
    end
    [X, ~] = scaling(X, [], "minmax");
    y = zeros(1,numel(Ks));
    i = 1;
    for k = Ks
        [cluster_idx, C, sumd] = kmeans(X,k);
        y(i) = sum(sumd);
        i=i+1;
    end
    figure
    plot(Ks, y)
    ylabel("SSE")
    xlabel("Number of Clusters")
end

