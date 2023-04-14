function [X, ca_Y] = get_train_test_data(...
    DATASET, train_size, descriptors, testing_images, color, probabilities)
%GET_TRAIN_TEST_DATA 

arguments
    DATASET {string};
    train_size = 0.8;
    descriptors = [Descriptor.Roughness Descriptor.Color ];
    testing_images = [68, 137, 143];
    color = true;
    probabilities = false;
end

if strcmp(DATASET, 'Dataset2')
    %ca = matrix2ca('Dataset2/Descriptors/');
    %ca = matrix2ca('Dataset2/Descriptors512GLRLM/');
    ca = matrix2ca('Dataset2/DescriptorsProbability/');
    %ca = matrix2ca('Dataset2/DescriptorsProbabilityColorGLCMGLRL/');
    n = numel(ca);
    n_train = floor(n * train_size); % Training set size
    n_train = 10;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    %ca_Y = ca(1:n_train); % test on training data
    %ca_Y = ca(n_train+1:n); % test on testing data
    ca_Y = ca(n_train+1:n_train+5); % test on testing data
    
elseif strcmp(DATASET, 'Dataset256')
    %ca = matrix2ca('Dataset/Descriptors256/'); % probability
    %ca = matrix2ca('Dataset/Descriptors256_new_color/'); % probability
    %ca = matrix2ca('Dataset/SmallImagesDescriptors/');
    ca = matrix2ca('Dataset/Descriptors256_binary/'); %binary
    n = numel(ca);
    n_train = floor(n * train_size); % Training set size
    %n_train = 1;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    X = under_sample(X);
    %ca_Y = ca(1:n_train); % test on training data
    %ca_Y = ca(n_train+1:n); % test on testing data
    ca_Y = ca(n_train+1:n_train+5); % test on testing data
    
elseif strcmp(DATASET, 'Segmentation')
    X = get_descriptors(load_images(68), descriptors, color);
    ca_Y = [];
    
elseif strcmp(DATASET, 'DatasetSelection')    
    %Descriptors folder:
    % Descriptors (16), Descriptors8
    patch_size = 'Descriptors'; 
    
    %Ground truth:
    % GroundTruthBinary, GroundTruthProbability, GroundTruthBinaryCropped
    str_gt = 'GroundTruthBinaryCropped'; 
    gt_ca = descriptor2ca(sprintf('DatasetSelection/%s/%s/',patch_size, str_gt));
    gt = cellfun(@(x) x.X, gt_ca, 'UniformOutput', false);
    idxs = cellfun(@(x) x.I, gt_ca, 'UniformOutput', false);
    
    %Descriptor(s):
    % LBP, LBP_HSV, LBP_RGB,
    % StatMomHSV, StatMomHSV34, StatMomRGB,
    % GLCM_HSV, GLCM_RGB, GLRLM, GLCMGray1, GLCMGray7
    
    str_descriptor = 'LBP_HSV'; 
    desc1 = descriptor2ca(sprintf('DatasetSelection/%s/%s/', patch_size, str_descriptor));
    desc1 = cellfun(@(x) x.X, desc1, 'UniformOutput', false);
%     str_descriptor = 'GLCM_HSV';
%     desc2 = descriptor2ca(sprintf('DatasetSelection/%s/%s/', patch_size, str_descriptor));
%     desc2 = cellfun(@(x) x.X, desc2, 'UniformOutput', false);
    
    % Concatenate ground truth with descriptor(s)
    ca_hcat = horzcat(desc1, gt);
    X = cellfun(@(x,y) [x,y], ca_hcat(:,1), ca_hcat(:,2), 'UniformOutput', false);
%     ca_hcat = horzcat(desc1, desc2, gt);
%     X = cellfun(@(x,y,z) [x,y,z], ca_hcat(:,1), ca_hcat(:,2), ca_hcat(:,3), 'UniformOutput', false);
    
    % Convert to universal format "cell(struct(X,I))"
    n = numel(X);
    ca = cell(n,1);
    for idx = 1:n
        img.X = X{idx};
        img.I = idxs{idx};
        ca{idx} = img;
    end
    ca = ca(randperm(numel(ca))); % shuffle
    
    % Train-Test Split
    n = numel(ca);
    n_train = floor(n * train_size); % Training set size
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    
    X = under_sample(X);
    
    %ca_Y = ca(1:n_train); % test on training data
    ca_Y = ca(n_train+1:n); % test on testing data
    %ca_Y = ca(n_train+1:n_train+5); % test on testing data
    
else
    X = get_descriptors(load_images(), descriptors, color, probabilities);
    %X = matfile('X10.mat').X;
    
    n = numel(testing_images);
    ca_Y = cell(n,1);
    
    for i=1:n
        Y.X = get_descriptors(load_images(testing_images(i)), descriptors, color, probabilities); 
        Y.I = testing_images(i);
        ca_Y{i} = Y;
    end
end

end

function X_ = under_sample(X)
    n_ones = sum(X(:,end));
    
    X_ = zeros(2*n_ones,size(X,2));
    for i = 0:1
        G = X(X(:,end)==i,:) ; % Group data
        idx = randperm(size(G,1),n_ones) ;
        X_(i*n_ones+1:(i+1)*n_ones,:) = G(idx, :);
    end
    
    rperm = randperm(size(X_,1),size(X_,1));
    X_ = X_(rperm, :);
end

function ca = descriptor2ca(folder)
    listing = dir(folder);
    ca = cell(numel(listing)-2, 1);
    for i = 3:numel(listing)
        X_save = load([folder listing(i).name]);
        M.X = X_save.X;
        M.I = sscanf(listing(i).name,'X%d'); %i-3;
        ca{i-2} = M;
    end
end

