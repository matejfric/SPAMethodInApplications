function [ca] = load_images(numbers, DATASET)
%LOAD_IMAGES 
% Inputs:
%     numbers...vector of numbers corresponding to images to be loaded
% Outputs:
%     ca........cell array with images
arguments
    numbers = [ 172, 177, 179, 203, 209, 212, 228, 240 ]
    DATASET = 'Dataset'
end

switch nargin
    case 0
        if strcmp(DATASET, 'Dataset') 
            numbers = [ 172, 177, 179, 203, 209, 212, 228, 240 ]; % small images
        else
            numbers = 1:5;
        end
end

n = length(numbers);
ca = cell(n,2);

for i = 1 : n
    if strcmp(DATASET, 'Dataset') 
        ca{i,1} = imread(sprintf('%s/Original/%d.jpg', DATASET, numbers(i)));
    else
        ca{i,1} = imread(sprintf('%s/Original/%d.jpeg', DATASET, numbers(i)));
    end
    ca{i,2} = imread(sprintf('%s/Annotations/%d.png', DATASET, numbers(i)));
end
    
end

