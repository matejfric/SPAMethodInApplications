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

[width, height] = get_screen_resolution();

figure('Renderer', 'painters', 'Position', [width/4 height/4 width/2 height/2])
sgtitle(caption) 
subplot(2,2,1)
imshow(cell2mat(original_rgb))
subplot(2,2,2)
imshow(annnotation, [])
subplot(2,2,3)
imshow(ground_truth, [])
subplot(2,2,4)
imshow(prediction, [])
colorbar

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

function [width, height] = get_screen_resolution()
    set(0,'units','pixels')
    res = get(0,'ScreenSize');
    res = res(3:4);
    width = res(1);
    height = res(2);
end
