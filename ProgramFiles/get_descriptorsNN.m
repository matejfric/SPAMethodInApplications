function [X] = get_descriptorsNN(ca_dataset)
%GET_DESCRIPTORS Performs roughness and color analysis

nchannels = 3;

net = resnet18;
sz = net.Layers(1).InputSize; % size of the first layer

% get descriptors
layer_name = 'pool5';

all_patches = {};

[images, ~] = size(ca_dataset);

for img = 1:images
    I = im2double(ca_dataset{img, 1}); % RGB
%    I(:,:,end+1:end+3) = im2double(rgb2hsv(ca_dataset{img, 1})); % HSV
    
    % Patches for each color channel
    ca_patches = cell(1, nchannels);
    for c = 1:nchannels
        [ca_patches{c}, ~] = patchify(I(:,:,c));
    end
    
    [rows, cols] = size(ca_patches{1});
    
    for i=1:rows
        for j=1:cols
            mypatch = zeros(16,16,nchannels);
            for c=1:nchannels
                mypatch(:,:,c) = ca_patches{c}{i,j};
            end
            
            % Patch color descriptor
            all_patches{end+1} = imresize(mypatch,sz(1:2));
        end
    end
    
end

X = activations(net,cell2table(all_patches'),layer_name,'OutputAs','rows');

X_True = get_ground_truth(ca_dataset, false);

X = [X, X_True];

end

