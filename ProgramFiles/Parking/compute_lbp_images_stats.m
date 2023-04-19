function [X] = compute_lbp_images_stats(ca_images)
%COMPUTE_LBP 

n_images = length(ca_images);
n_bins = 256;

% Matrix of descriptors
X = zeros(n_images, 6);

fprintf("LBP processing...")

for i = progress(1:n_images)
    I = im2double(rgb2gray(ca_images{i}.X));
    effLBP = efficientLBP(I);
    
    [patch_hist,intensity] = compute_histogram(effLBP);
                
    mu = dot(patch_hist,intensity); % Mean
    sigma = sqrt(dot((intensity - mu).^2,patch_hist)); % Standard deviation
    delta = dot((intensity - mu).^3,patch_hist)/(sigma^3); % Skewness
    nu = dot((intensity - mu).^4,patch_hist)/(sigma^4); % Kurtosis
    rho = -dot(patch_hist(patch_hist > 0),log(patch_hist(patch_hist > 0)))/log(2); % Entropy
    % Notice that: lim_{x->0+}(x*log(x)) = |l.H.| = 0 
    myrange = max(max(I)) - min(min(I)); % Range

    % Patch descriptor for channel 'c'
    X(i,:) = [mu, sigma, delta, nu, rho, myrange];
end

end

function [myhist,intensity] = compute_histogram(mypatch)
    nbins = 256;
    [N,edges] = histcounts(mypatch,nbins);
    myhist = N/sum(N);
    intensity = 0.5 * (edges(1:end-1) + edges(2:end));
end

