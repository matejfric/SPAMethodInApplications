function [f] = visualize(original_rgb, annnotation, ground_truth, prediction, caption)
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

%[width, height] = get_screen_resolution();
%figure('Renderer', 'painters', 'Position', [width/4 height/4 width/2 height/2])

f=figure;
f.WindowState = 'maximized';
sgtitle(caption) 
fontSize = 15;

tiledlayout(1,5,'TileSpacing','Compact','Padding','Compact');
% Original image
nexttile
imshow(add_border(cell2mat(original_rgb)))
title("Original image", 'Interpreter', 'latex', 'FontSize', fontSize)
% Ground truth
nexttile
imshow(add_border(annnotation), [])
box on
title("Ground truth", 'Interpreter', 'latex', 'FontSize', fontSize)
% Ground truth converted to patches
nexttile
imshow(add_border(ground_truth), [])
box on
title("Ground truth patches", 'Interpreter', 'latex', 'FontSize', fontSize)
% Prediction
nexttile
imshow(add_border(prediction), [])
box on
title("Prediction", 'Interpreter', 'latex', 'FontSize', fontSize)
colorbar
% Prediction overlay
nexttile
irig_img = cell2mat(original_rgb);
% Crop
sz = size(irig_img(:,:,1));
ms = mod(sz,16);
sz = sz - ms; 
irig_img = irig_img(1:sz(1),1:sz(2),:);
imshow(add_border(irig_img));
box on
title("Prediction overlay", 'Interpreter', 'latex', 'FontSize', fontSize)
red = cat(3, ones(sz), zeros(sz), zeros(sz));
hold on
h = imshow(red);
set(h, 'AlphaData', 0.65 .* prediction(1:sz(1),1:sz(2)))

if false
    % Original image
    subplot(n_rows,n_cols,1)
    imshow(cell2mat(original_rgb))
    % Ground truth
    subplot(n_rows,n_cols,2)
    imshow(annnotation, [])
    % Ground truth converted to patches
    subplot(n_rows,n_cols,3)
    imshow(ground_truth, [])
    % Prediction
    subplot(n_rows,n_cols,4)
    imshow(prediction, [])
    colorbar
    % Prediction overlay
    subplot(n_rows,n_cols, 5)
    irig_img = cell2mat(original_rgb);
    imshow(irig_img);
    sz = size(irig_img(:,:,1));
    red = cat(3, ones(sz), zeros(sz), zeros(sz));
    hold on
    h = imshow(red);
    set(h, 'AlphaData', 0.75 .* prediction(1:sz(1),1:sz(2)))
end

end

function img = add_border(img)
%ADD_BORDER
    % Define border width in pixels
    border_width = 3;
    % Define border color
    border_color = 0;
    % Add the border using the padarray() function
    img = padarray(img, [border_width, border_width], border_color, 'both');
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
