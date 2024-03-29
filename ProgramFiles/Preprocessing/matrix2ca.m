function [ca] = matrix2ca(folder)
%MATRIX2CA Loads individual matrices of descriptors from disk to cell array.

listing = dir(folder);
%X = [];
ca = cell(numel(listing)-2, 1);
for i = 3:numel(listing)
    
    X_save = load([folder listing(i).name]);
    
    M.X = X_save.X;
    M.I = sscanf(listing(i).name,'X%d'); %i-3;
    ca{i-2} = M;
    
    %X = [X; X_save.X]; % Load into one matrix
    %ca{i-2} = X_save.X;

end

%ca = ca(randperm(numel(ca))); % shuffle

end
