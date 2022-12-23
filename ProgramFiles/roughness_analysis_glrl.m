function [X, features] = roughness_analysis_glrl(ca_dataset)
%ROUGHNESS_ANALYSIS Extract GLRL statistics per patch

addpath('ProgramFiles/GLRL');

warning('off')

[images, ~] = size(ca_dataset);

row = 1;

features = 4*11;

X = zeros(1e5, features);

for img = progress(1:images)
    I = im2double(rgb2gray(ca_dataset{img, 1}));
    
    [ca_patches, ~] = patchify(I);
    
    [rows, cols] = size(ca_patches);
    
    for i=1:rows
        for j=1:cols
            % Horizonatal GLCM with 32 grey levels and distance=7
            %! (A lot of hyperparameters)
            glrlm = grayrlmatrix(ca_patches{i,j});
            stats = reshape(grayrlprops(glrlm),1,[]);
            %! Are 4 features enough?
            X(row,1:features) = stats;
            row = row + 1;
        end
    end
end

X = X(1:row-1, :); % Crop to non-null rows.
X(isnan(X))=0; % NaN => 0

end

