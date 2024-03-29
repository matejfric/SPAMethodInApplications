function [X, features] = color_analysis_wo_range(ca_dataset)
%COLOR_ANALYSIS Extract color features per patch 

%fprintf("Performing color analysis...\n");

[images, ~] = size(ca_dataset);

nchannels = 6;
features = nchannels * 5; % 6 features for each color channel
X = zeros(1e5, features);

row = 1;

for img = 1:images
    I = im2double(ca_dataset{img, 1}); % RGB
    I(:,:,end+1:end+3) = im2double(rgb2hsv(ca_dataset{img, 1})); % HSV
    
    % Patches for each color channel
    ca_patches = cell(1, nchannels);
    for c = 1:nchannels
        [ca_patches{c}, ~] = patchify(I(:,:,c));
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
                
                % Patch descriptor for channel 'c'
                Xc{c} = [mu;sigma;delta;nu;rho];
            end
            
            % Patch color descriptor
            X(row,1:features) = [Xc{1};Xc{2};Xc{3};Xc{4};Xc{5};Xc{6}];
            
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
