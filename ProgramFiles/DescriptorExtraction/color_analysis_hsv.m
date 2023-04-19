function [X, features] = color_analysis_hsv(ca_dataset)
%COLOR_ANALYSIS Extract color features per patch 

%fprintf("Performing color analysis...\n");

[images, ~] = size(ca_dataset);

nchannels = 3; % HSV
features = nchannels * 6; % 6 features for each color channel
X = zeros(1e5, features);

row = 1;

for img = 1:images
    I = im2double(rgb2hsv(ca_dataset{img, 1})); % HSV
    
    % Patches for each color channel
    ca_patches = cell(1, nchannels);
    for c = 1:nchannels
        [ca_patches{c}, ~] = patchify(I(:,:,c),16,false);
    end
    
    [rows, cols] = size(ca_patches{1});
    
    for i=1:rows
        for j=1:cols
            Xc = cell(1,nchannels);
            for c=1:nchannels
                mypatch = ca_patches{c}{i,j};
                [patch_hist,intensity] = compute_histogram(mypatch);
                
                mu = dot(patch_hist,intensity); % Mean
                sigma = sqrt(dot((intensity - mu).^2,patch_hist)); % Standard deviation
                delta = dot((intensity - mu).^3,patch_hist)/(sigma^3); % Skewness
                nu = dot((intensity - mu).^4,patch_hist)/(sigma^4); % Kurtosis
                rho = -dot(patch_hist(patch_hist > 0),log(patch_hist(patch_hist > 0)))/log(2); % Entropy
                % Notice that: lim_{x->0+}(x*log(x)) = |l.H.| = 0 
                myrange = max(max(mypatch)) - min(min(mypatch)); % Range
                
                % Patch descriptor for channel 'c'
                Xc{c} = [mu, sigma, delta, nu, rho, myrange];
            end
            
            % Patch color descriptor
            X(row,1:features) = cell2mat(Xc(1:nchannels));
            
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


