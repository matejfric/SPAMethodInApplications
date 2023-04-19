function [X, ca_Y] = get_data_from_dataset_selection(train_size, ground_truth, descriptors)
%GET_DATA_FROM_DATASET_SELECTION Summary of this function goes here
arguments
    train_size = 0.8;
    
    ground_truth = 'GroundTruthBinaryCropped';
    % GroundTruthBinary, GroundTruthProbability, GroundTruthBinaryCropped
    
    descriptors = ["LBP_HSV" "StatMomHSV"]
    % LBP, LBP_HSV, LBP_RGB,
    % StatMomHSV, StatMomHSV34, StatMomRGB,
    % GLCM_HSV, GLCM_RGB, GLRLM, GLCMGray1, GLCMGray7
end
    %Descriptors folder:
    % Descriptors (16), Descriptors8
    desc_folder = 'Descriptors'; 
    
    gt_ca = descriptor2ca(sprintf('DatasetSelection/%s/%s/',desc_folder, ground_truth));
    gt = cellfun(@(x) x.X, gt_ca, 'UniformOutput', false);
    idxs = cellfun(@(x) x.I, gt_ca, 'UniformOutput', false);
    
    if length(descriptors) == 1
        desc = descriptor2ca(sprintf('DatasetSelection/%s/%s/', desc_folder, descriptors(1)));
        ca_hcat = cellfun(@(x) x.X, desc, 'UniformOutput', false);
    else
        ca_desc = cell(length(descriptors),1);
        for d = 1:length(descriptors)
            str_desc = descriptors(d); 
            desc = descriptor2ca(sprintf('DatasetSelection/%s/%s/', desc_folder, str_desc));
            ca_desc{d} = cellfun(@(x) x.X, desc, 'UniformOutput', false);
            if d==2
                ca_hcat = horzcat(ca_desc{d-1}, ca_desc{d});
            end
            if d>2
                ca_hcat = horzcat(ca_hcat, ca_desc{d});
            end
        end
    end
    ca_hcat = horzcat(ca_hcat, gt);
    
    n_rows = size(ca_hcat,1);
    ca_merged = cell(n_rows,1);
    for row = 1:n_rows
        % Use horzcat to merge the columns of the current row
        ca_merged{row} = horzcat(ca_hcat{row,:});
    end
    
    % Convert to universal format "cell(struct(X,I))"
    n = length(ca_merged);
    ca = cell(n,1);
    for idx = 1:n
        img.X = ca_merged{idx};
        img.I = idxs{idx};
        ca{idx} = img;
    end
    ca = ca(randperm(numel(ca))); % shuffle
    
    % Train-Test Split
    n = numel(ca);
    n_train = floor(n * train_size); % Training set size
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    
    %X = under_sample(X);
    
    %ca_Y = ca(1:n_train); % test on training data
    ca_Y = ca(n_train+1:n); % test on testing data
    %ca_Y = ca(n_train+1:n_train+5); % test on testing data

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

