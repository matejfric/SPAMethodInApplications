function [X, features] = lbp_analysis16(ca_dataset)
%COLOR_ANALYSIS Extract LBP features per patch 

%fprintf("Performing LBP analysis...\n");

[images, ~] = size(ca_dataset);

X = zeros(1e5, 16); %TODO

row = 1;

for img = 1:images
    I = im2double(rgb2gray(ca_dataset{img, 1}));
    [ca_patches, ~] = patchify(I);
    [rows, cols] = size(ca_patches);
    
    for i=1:rows
        for j=1:cols
            mypatch = ca_patches{i,j};
            effLBP = efficientLBP(mypatch);
            
            [patch_hist,~] = compute_histogram(effLBP);

            % Patch descriptor for channel 'c'
            X(row,:) = patch_hist;
            
            row = row + 1;
        end
    end
end

X = X(1:row-1, :); % Crop to non-null rows.
X(isnan(X)) = 0;

end

function [myhist,intensity] = compute_histogram(mypatch)
    nbins = 16;
    [N,edges] = histcounts(mypatch,nbins);
    myhist = N/sum(N);
    intensity = 0.5 * (edges(1:end-1) + edges(2:end));
end
