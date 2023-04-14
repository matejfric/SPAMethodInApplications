function [X, features] = lbp_analysis(ca_dataset)
%COLOR_ANALYSIS Extract LBP features per patch 

%fprintf("Performing LBP analysis...\n");

[images, ~] = size(ca_dataset);

X = zeros(1e5, 6); %TODO

row = 1;

for img = 1:images
    I = im2double(rgb2gray(ca_dataset{img, 1}));
    [ca_patches, ~] = patchify(I);
    [rows, cols] = size(ca_patches);
    
    for i=1:rows
        for j=1:cols
            mypatch = ca_patches{i,j};
            effLBP = efficientLBP(mypatch);
            
            [patch_hist,intensity] = compute_histogram(effLBP);
            
            mu = dot(patch_hist,intensity); % Mean
            sigma = sqrt(dot((intensity - mu).^2,patch_hist)); % Standard deviation
            delta = dot((intensity - mu).^3,patch_hist)/(sigma^3); % Skewness
            nu = dot((intensity - mu).^4,patch_hist)/(sigma^4); % Kurtosis
            rho = -dot(patch_hist(patch_hist > 0),log(patch_hist(patch_hist > 0)))/log(2); % Entropy
            % Notice that: lim_{x->0+}(x*log(x)) = |l.H.| = 0 
            myrange = max(max(mypatch)) - min(min(mypatch)); % Range

            % Patch descriptor for channel 'c'
            X(row,:) = [mu, sigma, delta, nu, rho, myrange];
            
            row = row + 1;
        end
    end
end

X = X(1:row-1, :); % Crop to non-null rows.
X(isnan(X)) = 0;

end

function [myhist,intensity] = compute_histogram(mypatch)
    nbins = 256;
    [N,edges] = histcounts(mypatch,nbins);
    myhist = N/sum(N);
    intensity = 0.5 * (edges(1:end-1) + edges(2:end));
end
