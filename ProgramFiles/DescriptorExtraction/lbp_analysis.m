function [X, features] = lbp_analysis(ca_dataset)
%COLOR_ANALYSIS Extract LBP features per patch 

%fprintf("Performing LBP analysis...\n");

[images, ~] = size(ca_dataset);

X = zeros(1e5, 256); %TODO

row = 1;

for img = 1:images
    I = im2double(rgb2gray(ca_dataset{img, 1}));
    [ca_patches, ~] = patchify(I);
    [rows, cols] = size(ca_patches);
    
    for i=1:rows
        for j=1:cols
            mypatch = ca_patches{i,j};
            nFiltSize=8;
            nFiltRadius=1;
            filtR=generateRadialFilterLBP(nFiltSize, nFiltRadius);
            effLBP   = efficientLBP(mypatch, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
            effRILBP = efficientLBP(mypatch, 'filtR', filtR, 'isRotInv', true,  'isChanWiseRot', false);
            uniqueRotInvLBP=findUniqValsRILBP(nFiltSize);
            tightValsRILBP=1:length(uniqueRotInvLBP);
            % Use this function with caution- it is relevant only if 'isChanWiseRot' is false, or the
            % input image is single-color/grayscale
            effTightRILBP=tightHistImg(effRILBP, 'inMap', uniqueRotInvLBP, 'outMap', tightValsRILBP);

            binsRange=(1:2^nFiltSize)-1;
%             binsRange = 0:16:255;
%             x=hist(single( effLBP(:) ), binsRange);
%             y=hist(single( effRILBP(:) ), binsRange);
            %h=hist(single( effTightRILBP(:) ), tightValsRILBP);
            
            h = histcounts(effRILBP,16,'Normalization', 'probability');
            
%             nbins = 16;
%             [xx] = histcounts(effLBP(:),nbins);
%             [yy] = histcounts(effRILBP(:),nbins);
%             [zz] = histcounts(effTightRILBP(:),nbins);
            
            % Patch color descriptor
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
