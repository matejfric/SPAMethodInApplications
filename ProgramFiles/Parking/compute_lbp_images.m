function [X] = compute_lbp_images(ca_images)
%COMPUTE_LBP 

n_images = length(ca_images);
n_bins = 256;

% Matrix of descriptors
X = zeros(n_images, n_bins);

fprintf("LBP processing...")

for i = progress(1:n_images)
    I = im2double(rgb2gray(ca_images{i}.X));
    effLBP = efficientLBP(I);
    h = histcounts(effLBP,n_bins,'Normalization', 'probability');
    X(i,:) = h;
end

end

function [myhist,intensity] = compute_histogram(mypatch)
    nbins = 256;
    [N,edges] = histcounts(mypatch,nbins);
    myhist = N/sum(N);
    intensity = 0.5 * (edges(1:end-1) + edges(2:end));
end

