function [X, features] = roughness_analysis(ca_dataset)
%ROUGHNESS_ANALYSIS Extract GLCM statistics per patch

%! Try this from: https://www.mathworks.com/matlabcentral/answers/216708-convert-image-into-patches-of-size-64-64-and-get-each-patch#comment_607908
% glcm_output = blockproc(rgb2gray(ca_dataset{1,1}),...
%     [16, 16],...
%     @(block_struct) graycomatrix(...
%     block_struct.data, 'Offset', [0 1;-1 1;-1 0;-1  -1], 'NumLevel', 8, 'Symmetric',true), 'TrimBorder', false);

%fprintf("Performing roughness analysis...\n");

[images, ~] = size(ca_dataset);

row = 1;

offsets = [0 7]; %Khayatazad
%offsets = [0 7; -7 7;-7 0;-7 -7]; % +1% f1score

% 4 graycoprops * directions of GLCM calculation {0°, 45°, 90°, 135°}
features = 4 * size(offsets, 1);

X = zeros(1e5, features);

for img = 1:images
    I = im2double(rgb2gray(ca_dataset{img, 1}));
    
    [ca_patches, ~] = patchify(I);
    
    [rows, cols] = size(ca_patches);
    
    for i=1:rows
        for j=1:cols
            % Horizonatal GLCM with 32 grey levels and distance=7
            %! (A lot of hyperparameters)
            glcm = graycomatrix(ca_patches{i,j},...
                'NumLevels',32,...
                'Offset', offsets);
            stats = graycoprops(glcm);
            %! Are 4 features enough?
            X(row,1:features) = [stats.Contrast,...
                stats.Correlation,...
                stats.Energy,...
                stats.Homogeneity];
            
%             % Read annotations
%             patch = ca_annotatations{i,j};
%             % More than half of the pixels in the patch contain corrosion
%             %! (Another hyperparameter)
%             if sum(sum(patch)) / numel(patch) > 0.5 
%                 X(row, 5) = 1;
%             else
%                 X(row, 5) = 0;
%             end
            
            row = row + 1;
        end
    end
end

X = X(1:row-1, :); % Crop to non-null rows.
X(isnan(X))=0; % NaN => 0

end

