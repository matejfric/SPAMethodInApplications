function [X, ca_Y] = get_train_test_data(DATASET, descriptors, testing_images, color, probabilities)
%GET_TRAIN_TEST_DATA 

arguments
    DATASET {string};
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
    n_train = floor(n * 0.75); % Training set size
    n_train = 10;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    %ca_Y = ca(1:n_train); % test on training data
    %ca_Y = ca(n_train+1:n); % test on testing data
    ca_Y = ca(n_train+1:n_train+5); % test on testing data
    
elseif strcmp(DATASET, 'Dataset256')
    ca = matrix2ca('Dataset/Descriptors256/'); % probability
    %ca = matrix2ca('Dataset/Descriptors256_new_color/'); % probability
    %ca = matrix2ca('Dataset/SmallImagesDescriptors/');
    %ca = matrix2ca('Dataset/Descriptors256_binary/'); %binary
    n = numel(ca);
    %n_train = floor(n * 0.75); % Training set size
    n_train = 1;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    %ca_Y = ca(1:n_train); % test on training data
    %ca_Y = ca(n_train+1:n); % test on testing data
    ca_Y = ca(n_train+1:n_train+5); % test on testing data
    
elseif strcmp(DATASET, 'Segmentation')
    X = get_descriptors(load_images(68), descriptors, color);
    ca_Y = [];
    
elseif strcmp(DATASET, 'DatasetSelection')
    folder = 'DatasetSelection/Descriptors/';
    listing = dir(folder);
    ca = cell(numel(listing)-2, 1);
    for i = 3:numel(listing)
        X_save = load([folder listing(i).name]);
        M.X = X_save.X;
        M.I = sscanf(listing(i).name,'X%d'); %i-3;
        ca{i-2} = M;
    end
    ca = ca(randperm(numel(ca))); % shuffle
    
    n = numel(ca);
    n_train = floor(n * 0.75); % Training set size
    %n_train = 10;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    %ca_Y = ca(1:n_train); % test on training data
    ca_Y = ca(n_train+1:n); % test on testing data
    %ca_Y = ca(n_train+1:n_train+10); % test on testing data
    
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

