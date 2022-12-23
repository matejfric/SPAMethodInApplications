function [X, features] = color_analysis(ca_dataset)
%COLOR_ANALYSIS Extract color features per patch 

%TODO Compare performance with HSV

%fprintf("Performing color analysis...\n");

[images, ~] = size(ca_dataset);

features = 3*6; % 6 features for each color channel
X = zeros(1e5, features);

row = 1;

for img = 1:images
    I = ca_dataset{img, 1};
    
    % Patches for each color channel + ground truth
    ca_patches = cell(1,3);
    for c = 1:3
        [ca_patches{c}, ~] = patchify(I(:,:,c));
    end
    
    [rows, cols] = size(ca_patches{1});
    
    for i=1:rows
        for j=1:cols
            Xc = cell(1,3);
            for c=1:3
                mypatch = double(ca_patches{c}{i,j});
                [patch_hist,intensity] = compute_histogram(mypatch);
                
                mu = dot(patch_hist,intensity); % Mean
                sigma = sqrt(dot((intensity - mu).^2,patch_hist)); % Standard deviation
                delta = dot((intensity - mu).^3,patch_hist)/(sigma^3); % Skewness
                nu = dot((intensity - mu).^4,patch_hist)/(sigma^4); % Kurtosis
                rho = -dot(patch_hist(patch_hist > 0),log(patch_hist(patch_hist > 0)))/log(2); % Entropy
                % Notice that: lim_{x->0+}(x*log(x)) = |l.H.| = 0 
                myrange = max(max(mypatch)) - min(min(mypatch)); % Range
                
                % Patch descriptor for channel 'c'
                Xc{c} = [mu;sigma;delta;nu;rho;myrange];
            end
            
            % Patch color descriptor
            X(row,1:features) = [Xc{1};Xc{2};Xc{3}];
            
            row = row + 1;
        end
    end
end

X = X(1:row-1, :); % Crop to non-null rows.

X(isnan(X))=0; % NaN => 0

end

% Try this implementation:
% [counts, binCenters] = hist(data);
% P = counts/sum(counts);
% meanValue = (counts .* binCenters) / sum(counts);

function [myhist,intensity] = compute_histogram(mypatch)
    nbins = 256; % TODO
    edges = 0:256/nbins:256;
    [h,w] = size(mypatch);

    myhist = zeros(1,nbins);
    intensity = zeros(1,nbins);
    for i = 1:nbins
        if i==1
            myhist(i) = sum(sum(and(mypatch >= edges(i),mypatch <= edges(i+1))));
        else
            myhist(i) = sum(sum(and(mypatch > edges(i),mypatch <= edges(i+1))));
        end
        intensity(i) = 0.5*(edges(i) + edges(i+1));
    end
    myhist = myhist/(h*w);
end
