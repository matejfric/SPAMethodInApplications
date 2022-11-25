% Adapted from:
% https://matlab.fandom.com/wiki/Split_image_into_blocks

function [ca_patches, patch_size] = patchify(rgbImage, patch_size)
switch nargin
        case 2
        case 1
            %patch_size = 16; % Khaytazad
            patch_size = 16;
        otherwise
            warningMessage = sprintf('Warning: invalid argument count in "patchify"');
            uiwait(warndlg(warningMessage));
end
%==========================================================================
% Divide an image up into patches using mat2cell().
[rows, columns, numberOfColorBands] = size(rgbImage);

% Solves padding
if(~((rem(rows, patch_size) == 0) && (rem(columns, patch_size) == 0)))
    padding_rows = 0;
    padding_columns = 0;
    if(~(rem(rows, patch_size) == 0))
        padding_rows = patch_size - rem(rows, patch_size);
    end
    if(~(rem(columns, patch_size) == 0))
        padding_columns = patch_size - rem(columns, patch_size);
    end
    padding = [ padding_rows, padding_columns];
    %! Padding with zeros - perhaps 'replicate' might be better (another hyperparameter?)
    rgbImage = padarray(rgbImage, padding, 0, 'post');  
    [rows, columns, ~] = size(rgbImage);
end

% Figure out the size of each patch in rows.
% Most will be patch_size but there may be a remainder amount of less than that.
wholePatchRows = floor(rows / patch_size);
patchVectorR = [patch_size * ones(1, wholePatchRows)];
% Figure out the size of each patch in columns.
wholePatchCols = floor(columns / patch_size);
patchVectorC = [patch_size * ones(1, wholePatchCols)];

% Create the cell array, ca. 
% Each cell (except for the remainder cells at the end of the image)
% in the array contains a blockSizeR by blockSizeC by 3 color array.
% This line is where the image is actually divided up into blocks.
if numberOfColorBands > 1
    % It's a color image.
    ca_patches = mat2cell(rgbImage, patchVectorR, patchVectorC, numberOfColorBands);
else
    ca_patches = mat2cell(rgbImage, patchVectorR, patchVectorC);
end

end