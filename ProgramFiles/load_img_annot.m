function [original_rgb, annnotation] = load_img_annot(dataset,idx)
%LOAD_IMG_ANNOT Summary of this function goes here
%   Detailed explanation goes here
    if strcmp(dataset, 'Dataset') 
        original_rgb{1} = imread(sprintf('Dataset/Original/%d.jpg', idx));
        annnotation{1} = imread(sprintf('Dataset/Annotations/%d.png', idx));
    elseif strcmp(dataset, 'Dataset256')
        original_rgb{1} = imread(sprintf('Dataset/Original256/%d.jpg', idx));
        annnotation{1} = imread(sprintf('Dataset/Annotations256/%d.png', idx));
    elseif strcmp(dataset, 'DatasetSelection')
        original_rgb{1} = imread(sprintf('DatasetSelection/Original256/%d.jpg', idx));
        annnotation{1} = imread(sprintf('DatasetSelection/Annotations256/%d.png', idx));
    else
        original_rgb{1} = imread(sprintf('Dataset2/Original/%d.jpeg', idx));
        annnotation{1} = imread(sprintf('Dataset2/Annotations/%d.png', idx));
    end  
end

