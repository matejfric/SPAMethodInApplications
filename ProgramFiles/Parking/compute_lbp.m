function [X] = compute_lbp(ca_images)
%COMPUTE_LBP 

n_images = length(ca_images);
n_bins = 256;

% Matrix of descriptors
X = zeros(1e5, n_bins); %TODO

fprintf("LBP processing...")

row = 1;
for img = progress(1:n_images)
    I = im2double(rgb2gray(ca_images{img}.X));
    [ca_patches, ~] = patchify(I,20);
    [rows, cols] = size(ca_patches);
    
    for i=1:rows
        for j=1:cols
            mypatch = ca_patches{i,j};
            effLBP = efficientLBP(mypatch);
            h = histcounts(effLBP,n_bins,'Normalization', 'probability');
            X(row,1:length(h)) = h;
            row = row + 1;
        end
    end
end

X = X(1:row-1, :); % Crop to non-null rows.

X(isnan(X))=0; % NaN => 0

end

function [myhist,intensity] = compute_histogram(mypatch)
    nbins = 256;
    [N,edges] = histcounts(mypatch,nbins);
    myhist = N/sum(N);
    intensity = 0.5 * (edges(1:end-1) + edges(2:end));
end

