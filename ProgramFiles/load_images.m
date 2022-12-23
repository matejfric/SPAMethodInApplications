function [ca] = load_images(numbers)
%LOAD_IMAGES 
% Inputs:
%     numbers...vector of numbers corresponding to images to be loaded
% Outputs:
%     ca........cell array with images

DATASET = 'Dataset';

switch nargin
    case 0
        numbers = [ 172, 177, 179, 203, 209, 212, 228, 240 ]; % small images
end

    n = length(numbers);
    ca = cell(n,2);
    
    for i = 1 : n
        ca{i,1} = imread(sprintf('%s/Original/%d.jpg', DATASET, numbers(i)));
        ca{i,2} = imread(sprintf('%s/Annotations/%d.png', DATASET, numbers(i)));
    end
    
end

