function [img] = corrosion_overlay(original_rgb,prediction)
%CORROSION_OVERLAY Summary of this function goes here
%   Detailed explanation goes here
prediction = vector2image(prediction, original_rgb);
fontSize = 14;

% Prediction overlay
figure
irig_img = cell2mat(original_rgb);
% Crop
sz = size(irig_img(:,:,1));
ms = mod(sz,16);
sz = sz - ms; 
irig_img = irig_img(1:sz(1),1:sz(2),:);
imshow(irig_img);
title("Prediction overlay", 'Interpreter', 'latex', 'FontSize', fontSize)
red = cat(3, ones(sz), zeros(sz), zeros(sz));
hold on
h = imshow(red);
set(h, 'AlphaData', 0.65 .* prediction(1:sz(1),1:sz(2)))

frame = getframe(gcf);
img = frame2im(frame);

% Close the figure
close;

% Show the image (optional)
%imshow(img);

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


