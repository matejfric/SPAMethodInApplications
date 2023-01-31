function [] = visualize(original_rgb, annnotation, ground_truth, prediction, caption)
%VISUALIZE 
arguments
    original_rgb
    annnotation
    ground_truth
    prediction
    caption = ''
end

ground_truth = vector2image(ground_truth, original_rgb);
prediction = vector2image(prediction, original_rgb);
annnotation = cell2mat(annnotation);
annnotation(annnotation==1) = 255;   

figure
montage([original_rgb, annnotation, ground_truth, prediction],...
    "Size", [1 4], "BackgroundColor", "red", 'BorderSize', 5);
ax = gca;
ax.PositionConstraint = "outerposition";
title(caption);

end

function img = vector2image(vector, original_image)
    I = im2double(rgb2gray(cell2mat(original_image))); 
    [ca_patches, ~] = patchify(I);
    [patch_size, ~] = size(cell2mat(ca_patches(1,1)));
    [rows, cols] = size(ca_patches);
    
    for i=1:rows
        for j=1:cols
           ca_patches{i,j} = zeros(patch_size) + vector((i-1)*cols+j);
        end
    end
    
    img = cell2mat(ca_patches);
end

