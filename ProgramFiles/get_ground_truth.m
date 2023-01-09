function [ground_truth] = get_ground_truth(ca_dataset)
%GET_GROUND_TRUTH

[images, ~] = size(ca_dataset);

ground_truth = zeros(1e5, 1); % column of annotations (1...corroded, 0...not corroded)

row = 1;

for img = 1:images
    
    I = im2double(im2gray(ca_dataset{img, 2}));
    [ca_annotatations, ~] = patchify(I);
    [rows, cols] = size(ca_annotatations);
    
%     [ca_annotatations, ~] = patchify(ca_dataset{img, 2});
%     [rows, cols] = size(ca_annotatations);
    
    for i=1:rows
        for j=1:cols
            % Read annotations
            patch = ca_annotatations{i,j};
            % More than half of the pixels in the patch contains corrosion
            %! (Another hyperparameter)
            if sum(sum(patch>0)) / numel(patch) > 0.5 
                ground_truth(row) = 1;
            else
                ground_truth(row) = 0;
            end
            
            % % Probability that the patch contains corrosion
            % ground_truth(row) = sum(sum(patch>0)) / numel(patch);
            
            row = row + 1;
        end
    end
end

ground_truth = ground_truth(1:row-1); % Crop to non-null rows.

end
